---
title: Stochastic Progressive Photon Mapping In Alpha7XRenderer
date: 2024-10-06 16:39:58
tags:
---

# Introduction

I have written [a post explaining SPPM](https://blog.csdn.net/u010669231/article/details/104267262) and implemented a DXR version of photon mapping ([Screen Space Photon Mapping](https://blog.csdn.net/u010669231/article/details/117253961)) based on the ray tracing gems. The purpose of this post is to describe how SPPM is implemented in my tiny offline renderer using **embree**, Alpha7x that references **PBRT**. Before describing SPPM, we should first introduce some related algorithms: PM (basic photon mapping) and PPM (Progressive Photon Mapping).

## Basic Photon Mapping

Path tracing is impossible for the LS(S|D)*S path. A simple LSDS path is shown below, which includes a point light, a pinhole camera, and a diffuse surface behind a sheet of glass.

<p align="center">
    <img src="/resource/sppm/image/sds_example.png" width="60%" height="60%">
</p>

The glass is specular transmission. In this post, specular is absolute specular, in other words, its roughness is zero, which means only one input lighting direction contributes to the final light. And transmission means there is no contribution from reflection in our example.

There is only one light path that contributes to the given camera ray in this example, which is nearly impossible to be found. In path tracing, a ray is traced from a camera and first hits a glass surface. Since the glass is specular transmission, the direct lighting contribution is zero and the ray refracts to the other side of the glass. After refraction, the ray hits the diffuse material and generate a random ray to trace the input light. As we described before, only one direction is valid for the generated rays. The probability of selecting this particular ray is nearly zero.

The basic photon mapping algorithm solves this problem by resusing the nearyby photons, which introduces bias. Although it is a bias method, it could be consistent by increasing the sample number, which will be described later.

<p align="center">
    <img src="/resource/sppm/image/pm_reuse.png" width="60%" height="60%">
</p>

The basic photon mapping is a two pass algorithm:
1.Photons are first traced from the light source. Photon interactions are recorded on the surfaces of the scene and then organized in a spatial data structure (generally a KD-tree) for use at rendering time. 
2.A second pass follows paths starting from the camera; at each path vertex, nearby photons are used to estimate indirect illumination.

Considering that it is similar to SPPM, which will be discussed in more detail later, we will not delve into its details at this time. There is a limitation to the basic photon mapping method, in that the amount of memory available to store photons is limited, thus limiting the amount of photons that can be traced. In the path tracing algorithm, we could trace infinite rays, which means as the number of iterations increases, the rendering quality improves, but basic photon mapping rendering could not be improved once the memory limit was reached.

## Progressive Photon Mapping

The basic photon mapping algorithm stores the light path shot from the light source. As we discussed above, the light photons shot from the light are limited due to the memory limitation. Rather than storing the light path, progressive photon mapping addresses this issue by storing the camera path that traced from the camera.

This algorithm is also a two-pass method, but it differs from the basic PM algorithm in that it is inverse of it:
1.PPM traces the paths starting from the camera at first. Each pixel stores a representation of all of the non-specular path vertices found when generating paths that started in its extent. These vertices are named as visible points.
2.Traces photons from the light sources; at each photon-surface intersection, the photon contributes to the reflected radiance estimate for nearby visible points.

Please note that we only record the first diffuse hit point on the surface. When trace the camera starting from the camera, we calculate the direct lighting along this path.

In PPM the light transport equation could be decomposed into two separate integrals over direct{% katex %}L_d{% endkatex %} and indirect {% katex %}L_i{% endkatex %} incident radiance at each visible points.

{% katex %}L(p,w_o)=L_e(p,w_o)+\int_{S^2}f(p,w_o,w_i)L(p,w_i)|cos\theta_i|dw_i{% endkatex %}<br>
{% katex %}=L_e(p,w_o){% endkatex %}<br>
{% katex %}+\int_{S^2}f(p,w_o,w_i)L_d(p,w_i)|cos\theta_i|dw_i{% endkatex %}<br>
{% katex %}+\int_{S^2}f(p,w_o,w_i)L_i(p,w_i)|cos\theta_i|dw_i{% endkatex %}<br><br>

It will not be discussed further the terms emitted and direct lighting, which is similar to path tracing. The basic idea behind camera path search is to locate a visible point that can reuse nearby photons by preserving the photons traced from the light source on the surface of the visible point. Since diffuse surfaces are ideal for storing photons, we trace the path from the camera until it hits the diffuse surface.

In the basic photon mapping method, the photons are static after they are traced from the light source. The camera visible points are dynamic that change during iteration.

In progressive photon mapping, the camera visible points remain static after being traced by the camera. Photons are dynamic, as they increase during iteration.

The basic photon mapping algorithm is limited by its static part, photon number. Similar, the progressive photon mapping algorithm is also limited by its static part, camera visible points. For high resolution images or images that require many samples per pixel to resolve motion blur or depth of field, memory can still be a limitation.

The stochastic progressive photon mapping algorithm addresses these issues by converting camera visible points and photons into dynamic variables that could be changed during iteration.

# Stochastic Progressive Photon Mapping

Here is the main workflow of stochastic progressive photon mapping:

```cpp
for (int iter_idx = 0; iter_idx < iteration_num; iter_idx++)
{
	traceCameraPath(image_size, iter_idx);
	CSPPMGrid sppm_grid(image_area, camera_paths);
	sppm_grid.collectGridVisiblePointsList(camera_paths, image_size);
	generatePhotons(photons_per_iteration, iter_idx, sppm_grid);
	updateCameraVisiblePoints(image_size);
	std::cout << "iteration:" << iter_idx << std::endl;
}
```

For each iteration,

1.Trace rays from the camera, record the visible points on the first hit diffuse surface and calculated the direct lighting along the camera path found.
2.Construct the SPPM grid based on the visible points found in step 1.
3.Trace photons from the camera and calculate their contribution to nearby visible points along the light path.
4.Update visible points parameters, such as radius.

Before each iteration begins, the photons and visible point positions generated previously are discarded, which means the camera paths and the light paths are dynamic during each iteration.

## Trace Camera Path

The procedure for tracing the camera visible points is similar to path tracing. The visible points traced from the camera record its position at the first diffuse intersected surface and the output direction and the bsdf when it hits the diffuse surface. 
```cpp
struct SVisiblePoint
{
	glm::vec3 position;
	glm::vec3 wo;
	CBSDF bsdf;
	glm::vec3 beta;
};
```
The basic workflow is as follows:
1.Generate a ray that is traced from the camera.
```cpp
glm::vec3 ray_origin = camera->getCameraPos();
glm::vec2 pixel_sample_pos = glm::vec2(pix_pos) + sampler->getPixel2D();
glm::vec3 ray_direction = camera->getPixelRayDirection(pixel_sample_pos);
CRay ray(ray_origin, ray_direction);
```
2.Intersect the ray with the scene. Alpha7x uses embree to perform ray-scene intersection.
```cpp
const CSurfaceInterraction& sf_interaction = intersect(ray);
```
3.Sample and accumulate the direct lighting.
```cpp
glm::vec3 wo = -ray.direction;
cameral_path.l_d += SampleLd(sf_interaction, &bsdf, sampler) * beta;
```
4.If the intersection surface is diffused, stop tracing and record the visible point.
```cpp
EBxDFFlags bxdf_flag = bsdf.flags();
if (isDiffuse(bxdf_flag) || (isGlossy(bxdf_flag) && (depth == max_depth)))
{
	cameral_path.visible_point = SCameraPixelPath::SVisiblePoint{ sf_interaction.position,sf_interaction.wo, bsdf ,beta };
	find_visble_point = true;
}
```
5.Otherwise, generate a new ray randomly and update the path throughput.
```cpp
std::shared_ptr<SBSDFSample> bsdf_sample = bsdf.sample_f(wo, u, sampler->get2D());
```
6.Russian roulette.

Direct Lighting Result:
<p align="center">
    <img src="/resource/sppm/image/direct.png" width="55%" height="55%">
</p>

## SPPM Grid Construction

The contribution of a photon to nearby visible points should be calculated when it hits the scene, so we need an efficient data structure (for example, K-D tree) for finding nearby visible points. In alpha7x, we use hash grid to accelerate the nearby visible points search.

In each grid, there is a list of visible points that overlap the current grid.

```cpp
struct SGridVisiblePointsList
{
	SCameraPixelPath* cam_pixel_path;
	SGridVisiblePointsList* next_point = nullptr;
};
```
As visible points are located on the mesh surface, grids not located near the mesh surface do not have visible points. Therefore, most grids are sparsely filled. Our SPPM uses a hash function to transform the 3D grid index into a 1D array index, so we store the visible points in an array of visible points list`std::vector<SGridVisiblePointsList*> grids;`.

```cpp
class CSPPMGrid
{
public:
	CSPPMGrid(int ipt_image_area, const std::vector<SCameraPixelPath>& camera_paths);

	// ......
	// some public member functions

private:
	// ......
	// some private member functions

	glm::AABB grid_bound;
	std::allocator<SGridVisiblePointsList> node_allocator;
	std::vector<SGridVisiblePointsList*> grids;
	int grid_res[3];
	int image_area;
};
```
The grid bound is defined by the union of the visible points' positions in the scene, and the grid resolution is determined by calculating the maximum search radius of the visible points.

```cpp
CSPPMGrid::CSPPMGrid(int ipt_image_area, const std::vector<SCameraPixelPath>& camera_paths) :image_area(ipt_image_area)
{
	grids.resize(image_area);
	memset(grids.data(), 0, sizeof(SGridVisiblePointsList*) * image_area);

	float max_radius = 0.0;
	for (int idx = 0; idx < image_area; idx++)
	{
		const SCameraPixelPath& pixel = camera_paths[idx];
		grid_bound.extend(pixel.visible_point.position - glm::vec3(0.1, 0.1, 0.1));
		grid_bound.extend(pixel.visible_point.position + glm::vec3(0.1, 0.1, 0.1));
		max_radius = (std::max)(max_radius, pixel.radius);
	}

	glm::vec3 diagonal = grid_bound.getDiagonal();
	for (int i = 0; i < 3; ++i)
	{
		grid_res[i] = std::max<int>(std::ceil(diagonal[i] / max_radius), 1);
	}
}
```
Then, iterate over all of the visible points generated in the last step and add them to the corresponding grids.

```cpp
void CSPPMGrid::collectGridVisiblePointsList(std::vector<SCameraPixelPath>& camera_paths, const glm::u32vec2 image_size)
{
	for (glm::uint32 pixel_x = 0; pixel_x < image_size.x; pixel_x++)
	{
		for (glm::uint32 pixel_y = 0; pixel_y < image_size.y; pixel_y++)
		{
			int pixel_idx = pixel_x + pixel_y * image_size.x;
			SCameraPixelPath& pixel = camera_paths[pixel_idx];

			glm::vec3 vp_beta = pixel.visible_point.beta;
			if (vp_beta.x > 0 || vp_beta.y > 0 || vp_beta.z > 0)
			{
				float r = pixel.radius;
				glm::ivec3 p_min;
				glm::ivec3 p_max;
				getGridOffest(pixel.visible_point.position - glm::vec3(r, r, r), grid_bound, grid_res, p_min);
				getGridOffest(pixel.visible_point.position + glm::vec3(r, r, r), grid_bound, grid_res, p_max);

				for (int z = p_min.z; z <= p_max.z; z++)
				{
					for (int y = p_min.y; y <= p_max.y; y++)
					{
						for (int x = p_min.x; x <= p_max.x; x++)
						{
							uint32_t node_hash = hashVisPoint(glm::ivec3(x, y, z), image_area);
							SGridVisiblePointsList* pixel_node = node_allocator.allocate(1);
							pixel_node->cam_pixel_path = &pixel;
							pixel_node->next_point = grids[node_hash];
							grids[node_hash] = pixel_node;
						}
					}
				}
			}
		}
	}
}
```


There will be no contributions to the grid for camera pixels that do not find any visible points (for example, when the ray leaves the scene), which is path throughput equal to zero ({% katex %}\beta=0{% endkatex %}).
```cpp
glm::vec3 vp_beta = pixel.visible_point.beta;
if (vp_beta.x > 0 || vp_beta.y > 0 || vp_beta.z > 0)
{
	// add to the grid
}
```
<p align="center">
    <img src="/resource/sppm/image/radius.png" width="50%" height="50%">
</p>

>Given a visible point (filled circle) with search radius , the visible point is added to the linked list in all grid cells that the bounding box of the sphere of radius  overlaps. Given a photon incident on a surface in the scene (open circle), we only need to check the visible points in the voxel the photon is in to find the ones that it may contribute to.

```cpp
float r = pixel.radius;
glm::ivec3 p_min;
glm::ivec3 p_max;
getGridOffest(pixel.visible_point.position - glm::vec3(r, r, r), grid_bound, grid_res, p_min);
getGridOffest(pixel.visible_point.position + glm::vec3(r, r, r), grid_bound, grid_res, p_max);

for (int z = p_min.z; z <= p_max.z; z++)
	for (int y = p_min.y; y <= p_max.y; y++)
		for (int x = p_min.x; x <= p_max.x; x++)
			// add visible points
```
## Trace Photons

Next, trace photons from the light source. Here is the workflow.

1.Select a light from the scene randomly.
2.Sample an outgoing ray from the light. Based on the following equation, we can determine the initial path throughput:
{% katex %}\beta=\frac{|cos(w_{o})|L_e(p,w_o)}{p(light)p(l_{pos})p(l_{dir})}{% endkatex %}<br><br>

The probability of sampling this particular light is measured by {% katex %}p(light){% endkatex %}, while the probability of selecting this position on the light is measured by {% katex %}p(l_{pos}){% endkatex %}, and the probability of selecting this outgoing ray by {% katex %}p(l_{dir}){% endkatex %}.

```cpp
glm::vec3 beta = std::abs(glm::dot(light_normal, ray.direction)) * Le / (pdf_light * pdf_position * pdf_direction);
```
3.Trace the photon though the scene.

```cpp
for (int depth = 0; depth < max_depth; depth++)
{
	const CSurfaceInterraction& surface_iteraction = intersect(photon_ray);

	// do something
}
```

4.Calculate the photon's contribution to the near visible points when it hits the surface. Since the first intersection with the scene represents direct lighting which is already accounted for in the first step, we calculate the photon's contribution from the second intersection.

```cpp
if (surface_iteraction.is_hit == false)
{
	break;
}

if (depth > 0)
{
	sppm_grid.addPhoton(surface_iteraction.position, photon_ray.direction, beta);
}
```
A visible point records radiant flux using phi. The radiant flux of nearby visible points is accumulated when photons hit the surface.
The radiance is thw power emmited, reflected, transmitted or recieved by a surface, per solid angle, per projected unit area, which means:

{% katex %}L(p,w)=\frac{d^2\Phi(p,w)}{dwdAcos\theta}{% endkatex %}<br>
{% katex %}d^2\Phi(p,w)=L(p,w)cos\theta dwdA {% endkatex %}<br>
{% katex %}\Phi(p,w)=\int_{S^2}\int_{H^2}L(p,w)cos\theta dwdA{% endkatex %}

Its discrete representation is shown below, where {% katex %}\beta{% endkatex %} is the path throughput.

<p align="center">
    <img src="/resource/sppm/image/phi.png" width="39%" height="39%">
</p>

```cpp
void CSPPMGrid::addPhoton(glm::vec3 photon_pos, glm::vec3 photon_ray_dir, glm::vec3 beta)
{
	glm::ivec3 photon_grid_index;
	if (getGridOffest(photon_pos, grid_bound, grid_res, photon_grid_index))
	{
		int photon_hash_value = hashVisPoint(photon_grid_index, image_area);

		for (SGridVisiblePointsList* pixel_node = grids[photon_hash_value]; pixel_node != nullptr; pixel_node = pixel_node->next_point)
		{
			SCameraPixelPath* pixel = pixel_node->cam_pixel_path;
			float radius = pixel->radius;
			float photon_distance = glm::distance(pixel->visible_point.position, photon_pos);

			if (photon_distance * photon_distance > radius * radius)
			{
				continue;
			}

			glm::vec3 wi = -photon_ray_dir;
			glm::vec3 phi = beta * pixel->visible_point.beta * pixel->visible_point.bsdf.f(pixel->visible_point.wo, wi);
			pixel->phi += phi;
			pixel->m++;
		}
	}
}
```
Based on the above equation, every visible point in this grid whose distance to the photon is less than the radius should accumulate photon contributions. It should be noted that reusing nearby photons is incorrect mathematically, except when the radius is zero. This operation introduces the bias. The total number of contributing photons in this pass is stored in M.

5.Update the path throughput.
```cpp
glm::vec3 beta_new = beta * bsdf_sample->f * std::abs(glm::dot(bsdf_sample->wi, surface_iteraction.norm)) / bsdf_sample->pdf;
```
6.Russian roulette.

## Update Visible Points Parameters
### Radius Reduction
The PPM algorithm generates photon mapping progressively. Then, it estimates the density of the photon maps and combines them together:
{% katex %}d(x)=\frac{N}{\pi r^2} {% endkatex %}<br>

There is a problem with the radius being fixed in this case. As we mentioned above, the SPPM is a biased but consistent method when its gathering radius is towards zero. So it is critical to decrease the gathering radius during iteration.

After the first iteration, the photon density is shown in the following formula:

{% katex %}d_1(x)=\frac{N_1(x)}{\pi R_1(x)^2} {% endkatex %}<br>

And the photon density after the second iteration is:

{% katex %}d_2(x)=\frac{N_2(x)}{\pi R_2(x)^2} = \frac{N_2(x)}{\pi (R_1(x) - \Delta R_1(x))^2}{% endkatex %}<br>

If we trace enough photons, the density of photons in the first and second iterations is the same, which means that:
{% katex %}d_1(x)=d_2(x) {% endkatex %}<br>

and 
{% katex %}N_2(x)=\pi R_2(x)^2 d_2(x)=\pi(R_1(x) - \Delta R_1(x))^2 d_2(x){% endkatex %}<br>
<p align="center">
    <img src="/resource/sppm/image/alpha.png" width="65%" height="65%">
</p>

>Each hit point in the ray tracing pass is stored in a global data structure with an associated radius and accumulated photon power. After each photon tracing pass we find the new photons within the radius of each hit point, and we reduce the radius based on the **newly added photons**. The progressive radiance estimate ensures that the final value at each hit point will converge to the correct radiance value. For simplicity, we use a parameter {% katex %}\alpha {% endkatex %} = (0, 1) to control the fraction of photons to keep after every iteration.

Therefore,
{% katex %}N_2(x) = N_1(x) + \alpha M(x){% endkatex %}<br>

Then,

{% katex %}\pi(R_1(x) - \Delta R_1(x))^2 d_2(x) = N_2(x){% endkatex %}<br>
{% katex %}\pi(R_1(x) - \Delta R_1(x))^2 \frac{N_1(x) + M(x)}{\pi R(x)^2} = N_1  + \alpha M(x){% endkatex %}<br>
{% katex %}\Delta R_1(x) = R_1(x) - R_1(x) \sqrt{\frac{N_1(x) + \alpha M(x)}{N_1(x)+M(x)}}{% endkatex %}<br>
{% katex %}R_2(x) = R_1(x) - \Delta R_1(x) = R_1(x) \sqrt{\frac{N_1(x) + \alpha M(x)}{N_1(x)+M(x)}}{% endkatex %}<br>

In PBRT, the {% katex %}\alpha{% endkatex %} is 2/3:

```cpp
float gamma = 2.0 / 3.0;
float n_new = pixel.n + gamma * m;
float radius_new = pixel.radius * std::sqrt(n_new / (pixel.n + m));
```

### Flux Correction

The unnormalized total flux received for visible points is {% katex %}\tau{% endkatex %}. As we have generated a new radius for the next iteration, the {% katex %}\tau{% endkatex %} value should be corrected, since the radius is decreasing and some photons are outside the updated radius.

>One method for finding those photons would be to keep a list of all photons within the disc and remove those that are not within the reduced radius disc. However, this method is not practical as it would require too much memory for the photon lists.

For simplicity, we assume:
{% katex %}\frac{\tau_2(x)}{\tau_1(x) + \tau_m(x)} = \frac{area_2(x)}{area_1(x)} = \frac{R_2(x)^2}{R_1(x)^2}{% endkatex %}

```cpp
pixel.tau = (pixel.tau + pixel.phi) * radius_new * radius_new / (pixel.radius * pixel.radius);
```
### Radiance Evaluation

After each photon tracing pass we can evaluate the radiance at the hit points. Recall that the quantities stored include the current radius and the current intercepted flux multiplied by the BRDF. The evaluated radiance is multiplied by the pixel weight and added to the pixel associated with the hit point. To evaluate the radiance we further need to know the total number of emitted photons {% katex %}N_{emitted}{% endkatex %} in order to normalize τ (x, ~ω). The radiance is evaluated as follows
<p align="center">
    <img src="/resource/sppm/image/final.png" width="65%" height="65%">
</p>

# Result

Below is the SPPM result. Each pixel traces 200 samples and during each iteration traces 256 * 256 photons from the camera.

<p align="center">
    <img src="/resource/sppm/image/result.png" width="65%" height="65%">
</p>

[Project Source Code](https://github.com/ShawnTSH1229/alpha7x)

