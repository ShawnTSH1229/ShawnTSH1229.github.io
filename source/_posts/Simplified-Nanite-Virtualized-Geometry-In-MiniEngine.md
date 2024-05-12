---
title: Simplified Nanite In MiniEngine
date: 2024-05-08 22:09:35
index_img: /resource/simnanite/image/vis_clu_0.png
categories:
- My Projects
---
# Overview
todo
# Nanite Builder
## Overview
SimNanite (Simplified Nanite) building is an offline mesh processing during mesh import. It splits the mesh into clusters to provide a fine-grained mesh culling with a graph partition algorithm. Beyond the cluster level, SimNanite partitions the cluster into groups to accelerate mesh culling. In addtion, SimNanite simplifies the merged clusters at the current level rather than separate clusters in order to avoid the Lod crack artifact without boundnary edge locking.
<p align="center">
    <img src="/resource/simnanite/image/nanite_builder_overview.png" width="60%" height="60%">
</p>

## Triangle Partition

SimNanite partitions the mesh into clusters in order to perform fine-grained GPU culling. Meshes can be viewed as graphs, whose nodes are vertex points and edges are mesh topology. With the graph representation, we can partition it with the Metis graph library. 

Given a mesh without an index buffer, SimNanite processes the mesh triangles sequentially. For each triangle, SimNanite hash the vertex position to find the vertex index of the adjacent list array. Each vertex in the triangle has 6 edges. We add these edge points into the coresponding vertex adjacent unordered_set. 

<p align="center">
    <img src="/resource/simnanite/image/find_vertex_adjacent.png" width="50%" height="50%">
</p>

The Metis library's graph partition function input has two parts: vertex adjacent list array and vertex adjacent offsets. Vertex adjacent offsets record the offsets of each vertex in the vertex adjacent list.

<p align="center">
    <img src="/resource/simnanite/image/metis_input_format.png" width="60%" height="60%">
</p>

With the adjacent vertex list for each vertex, we can pack them into an array and record their offsets.

<p align="center">
    <img src="/resource/simnanite/image/metis_input.png" width="60%" height="60%">
</p>

After the triangle partition, Metis outputs an array containing the partition index of each vertex. The next step is to batch the triangles into clusters with the partition result and find out the linked clusters for each cluster. If the vertices of one triangle belong to different partitions, this triangle can be viewed as an "edge-cluster". We add these clusters to the linked clusters array in order to partition the clusters in the next step.

```cpp
idx_t tri_parts[3];
tri_parts[0] = part_a;

int triangle_part_num = 1;
if (part_b != part_a)
{
	tri_parts[triangle_part_num] = part_b;
	triangle_part_num++;
}

if ((part_c != part_a) && (part_c != part_b))
{
	tri_parts[triangle_part_num] = part_c;
	triangle_part_num++;
}

if (triangle_part_num == 2)
{
	out_clusters[tri_parts[0]].m_linked_cluster.insert(tri_parts[1]);
	out_clusters[tri_parts[1]].m_linked_cluster.insert(tri_parts[0]);
}
else if (triangle_part_num == 3)
{
	out_clusters[tri_parts[0]].m_linked_cluster.insert(tri_parts[1]);
	out_clusters[tri_parts[0]].m_linked_cluster.insert(tri_parts[2]);

	out_clusters[tri_parts[1]].m_linked_cluster.insert(tri_parts[0]);
	out_clusters[tri_parts[1]].m_linked_cluster.insert(tri_parts[2]);

	out_clusters[tri_parts[2]].m_linked_cluster.insert(tri_parts[0]);
	out_clusters[tri_parts[2]].m_linked_cluster.insert(tri_parts[1]);
}
```

## Cluster Partition

With clusters and linked clusters, we can partition them into cluster groups as we do in the triangle partition pass. In addition, we record the cluster group index for each vertex in order to link cluster groups between mesh LODs after mesh simplification. This is also the reason why we use the QEM mesh simplification approach rather than the voxel-based solution. It is difficult to map clusters between mesh LODs using the latter method.

<p align="center">
    <img src="/resource/simnanite/image/vtx_clu_grp_map.png" width="50%" height="50%">
</p>

Below is the cluster visualization at LOD level 0, 1 and 2 in SimNanite:

<p align="center">
    <img src="/resource/simnanite/image/vis_clu_0.png" width="50%" height="50%">
</p>

<p align="center">
    <img src="/resource/simnanite/image/vis_clu_1.png" width="50%" height="50%">
</p>

<p align="center">
    <img src="/resource/simnanite/image/vis_clu_2.png" width="50%" height="50%">
</p>

## Mesh Simplification

We simplify the mesh until the cluster number is less than 24 or fails to simplify the mesh. The mesh simplification library is Meshoptimizer. It employs the QEM method to simplify the mesh, which is similar to what the Unreal Engine does. Another important point is that we simplify the global mesh rather than the cluster, as the latter method causes the LOD crack whitout boundnary edge locking. 

<p align="center">
    <img src="/resource/simnanite/image/mesh_simplify.png" width="50%" height="50%">
</p>

## Build DAG

In the DAG (Directed Acyclic Graph) building pass, we organize the data and translate it into a GPU-friendly structure to acclerate to GPU-culling performed later. Each parent cluster group may link with one or more child cluster groups. Similarly, each child cluster group may be connected to one or more parent cluster groups.

<p align="center">
    <img src="/resource/simnanite/image/adg_structure.png" width="70%" height="70%">
</p>

SimNanite merges the resources of all lod level into a global resource array. That is to say, a simnanite mesh resource only has one vertex buffer, one index buffer, one cluster group array and one cluster array. UE's Nanite performs an additional compression process after mesh building. We ignore compression for simplicity.

<p align="center">
    <img src="/resource/simnanite/image/dag_data.png" width="50%" height="50%">
</p>

A nanite mesh resource contains several lod resources. Each LOD resource stores cluster group indices in the current level. The max cluster group number per LOD is 8. Cluster group structure contains two parts: child cluster group indices linked in the DAG structure and child cluster indices. Cluster is the minimum culling unit. It stores the vertex and index location in the mesh vertex buffer. Nanite mesh building consume a lot of time. To accelerate programming efficiency, we serialize the DAG structure on the disk and load it without building at the next launch.

```cpp
struct CSimNaniteClusterResource
{
	DirectX::BoundingBox m_bouding_box;
	
	int m_index_count;
	int m_start_index_location;
	int m_start_vertex_location;
};

struct CSimNaniteClusterGrpupResource
{
	DirectX::BoundingBox m_bouding_box;
	float cluster_next_lod_dist;
	int m_child_group_num;
	int m_cluster_num;

	std::vector<int>m_child_group_indices;
	std::vector<int>m_clusters_indices;

	typedef void (*group_serialize_fun)(char*, long long, void* streaming);
	void Serialize(group_serialize_fun func, void* streaming);
};

struct CSimNaniteLodResource
{
	int m_cluster_group_num;
	int m_cluster_group_index[8];
};

class CSimNaniteMeshResource
{
public:
	SSimNaniteMeshHeader m_header;

	DirectX::BoundingBox m_bouding_box;
	
	std::vector<DirectX::XMFLOAT3> m_positions;
	std::vector<DirectX::XMFLOAT3> m_normals;
	std::vector<DirectX::XMFLOAT2> m_uvs;
	std::vector<unsigned int> m_indices;

	std::vector<CSimNaniteClusterGrpupResource> m_cluster_groups;
	std::vector<CSimNaniteClusterResource>m_clusters;
	std::vector<CSimNaniteLodResource> m_nanite_lods;

	void LoadFrom(const std::string& source_file_an, const std::wstring& source_file, bool bforce_rebuild = false);
};
```

# Culling
## Instance Culling

This step performs instance-level GPU-Culling, which is easy to implement and not necessary to detail it. The input of this step is the scene instance data and the output is the instance culled by the camera.

scene instance data:
<p align="center">
    <img src="/resource/simnanite/image/ins_cull_scene.png" width="50%" height="50%">
</p>

instances viewed by camera:
<p align="center">
    <img src="/resource/simnanite/image/ins_cull_view.png" width="50%" height="50%">
</p>

instances culled by camera:
<p align="center">
    <img src="/resource/simnanite/image/ins_cull_scene_view.png" width="50%" height="50%">
</p>

## Persistent Culling

Unreal's Nanite build a BVH structure offline to accelerate the run-time cluster group culling. SimNanite removes the BVH culling and iterates the DAG cluster group naively.

Build Scene Vertex Buffer

Global Cluster Index Offset

Init Persistent Culling
1.group thread index 0, allocate the block






