---
title: Sky Atmosphere In XEngine
date: 2024-05-05 15:47:47
tags:
---
>**Note** that this [<u>**Project**</u>](https://github.com/lvcheng1229/XEnigine/blob/main/Source/Shaders/SkyAtmosphere.hlsl)  was written during my undergraduate studies. I am just organizing this project and publishing the blog for it at this time.

# Brief Introduction to UE's Implementation
This project is based on the implementation of the **unreal engine**. I assume that you have a basic knowledge of volume rendering and will **not go into detail** about it. If you are interested, you can read my blog about [<u>**Volume Rendering In Offline Rendering**</u>](https://shawntsh1229.github.io/2024/05/05/Volume-Rendering-In-Offline-Rendering/).

{% katex %}\sigma_{a}{% endkatex %}: absorption coefficient, the probability density that light is absorbed per unit distance traveled in the medium.<br><br>
{% katex %}\sigma_{s}{% endkatex %}: scattering coefficient, the probability of an out-scattering event occurring per unit distance.<br><br>
{% katex %}\sigma_{t}{% endkatex %}: attenuation coefficient, the sum of the absorption coefficient and the scattering coefficient, describes the combined effect of the absorption and out scattering.<br>
{% katex %}\sigma_{t} = \sigma_{a} + \sigma_{s}{% endkatex %}<br><br>
{% katex %}T(a,b){% endkatex %}: transmittance, the fraction of radiance that is transmitted between two points<br>
{% katex %}T(a,b)=e^{-\int_{a}^{b}\sigma_{t}(h)\mathrm{d}s}{% endkatex %}<br><br>
{% katex %}\phi_{u}{% endkatex %}: phase function, the angular distribution of scattered radiation at a point<br><br>
**Volume Rendering Euqation**:
The lighting result at the point x from direct v equals the combination of single scattering and multi-scattering:
{% katex %}L(x,v) = L_{1}(x,v) + L_{n}(x,v){% endkatex %}<br><br>
The single scattering result at point x from point p in direction v is the sum of the transmitted radiance reflected from the ground and the integration of the radiance from the view point to the ground or sky:
{% katex %}L_{1}(x,p,v) = T(x,p)L_{0}(p,v)+\int_{t=0}^{||p-x||}\sigma_{s}T(x,x-tv)T(x-tv,x-tv+t_{atmo}l_{i})\phi_{u}E_{i}\mathrm{d}t{% endkatex %}<br><br>
N-bounce multi-scattering ({% katex %}L_{n}{% endkatex %}) integrate the {% katex %}G_{n}{% endkatex %} in the point along the view direction, where {% katex %}G_{n}{% endkatex %} is the integration of the {% katex %}L_{n-1}{% endkatex %} scattering result from the direction on the sphere. Multi-Scattering is difficult to calculate compared to Single-Scattering. There are many simplifications that are made in UE's sky-atmosphere for multi-scattering, we will be introducing it later.
{% katex %}L_{n}(x,v) = \int_{t=0}^{||p-x||}\sigma_{s}(x)T(x,x-tv)G_{n}(x-tv,-v)\mathrm{d}t{% endkatex %}<br>
{% katex %}G_{n}(x,v)=\int_{\Omega}L_{n-1}(x,-v)\phi_{u}\mathrm{d}w{% endkatex %}
# Transmittance LUT

# Multi-Scattering
As we mentioned above, it's difficult to calculate the {% katex %}G_{n}{% endkatex %}, unreal engine makes two simplification for this part.
The first simplification is:
>Scattering events with order greater or equal to 2 are executed using an isotropic phase function {% katex %}\phi_{u}{% endkatex %}
{% katex %}G_{n}(x,v)=\int_{\Omega}L_{n-1}(x,-v)\phi_{u}\mathrm{d}w\approx \frac{1}{4\pi}\int_{\Omega}L_{n-1}(x,-v)\mathrm{d}t{% endkatex %}
The second simplification is:
>All points within the neighborhood of the position we currently shade receive the same amount of second order scattered light
{% katex %}L_{n-1}(x,v) = \int_{t=0}^{||p-x||}\sigma_{s}(x)T(x,x-tv)G_{n-1}(x-tv,-v)\mathrm{d}t\approx G_{n-1}\int_{t=0}^{||p-x||}\sigma_{s}(x)T(x,x-tv)\mathrm{d}t{% endkatex %}<br>
Combine these equation together, we get:
{% katex %}G_{n}(x,v)\approx \frac{1}{4\pi}\int_{\Omega}L_{n-1}(x,-v)\mathrm{d}w\approx \frac{1}{4\pi} G_{n-1}\int_{\Omega}\int_{t=0}^{||p-x||}\sigma_{s}(x)T(x,x-tv)\mathrm{d}t\mathrm{d}w{% endkatex %}

# Sky-View LUT

[<u>**source code can be found here.**</u>](https://github.com/ShawnTSH1229/XEngine)