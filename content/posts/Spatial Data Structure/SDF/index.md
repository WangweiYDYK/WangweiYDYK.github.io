+++
title = 'SDF'
date = 2025-02-22T13:38:09+08:00
draft = false
+++
# SDF

## 构建方式

## 基础存储格式

```cpp
class DistanceField3D
{
    // Lower left corner
    vec3f m_left;

    // grid spacing
    vec3f m_h;
    int nbx, nby, nbz;
    float* distance;
};
```

## 距离查询
预生成的三维有向距离场（SDF）中查询任意点的平滑距离值和法线方向  
距离采用三线性插值  
法线方向通过中心差分法估算SDF的梯度，梯度近似公式可以表示为
$$\nabla \mathrm{SDF} \approx\left(\frac{\partial d}{\partial x}, \frac{\partial d}{\partial y}, \frac{\partial d}{\partial z}\right)$$
```cpp
void DistanceField3D::getDistance(const vec3f& p, REAL& d, vec3f& normal) { // get cell and lerp values vec3f     
    fp = (p - m_left) * vec3f(1.0 / m_h[0], 1.0 / m_h[1], 1.0 / m_h[2]); 
    const int i = (int)floorf(fp[0]); 
    const int j = (int)floorf(fp[1]); 
    const int k = (int)floorf(fp[2]);
    if (i < 0 || i >= m_nbx - 1 || j < 0 || j >= m_nby - 1 || k < 0 || k >= m_nbz - 1)
    {
        d = 100000.0f;
        normal = vec3f(0, 0, 0);
        return;
    }
    vec3f ip = vec3f(i, j, k);

    vec3f alphav = fp - ip;
    REAL  alpha = alphav[0];
    REAL  beta = alphav[1];
    REAL  gamma = alphav[2];

    REAL d000 = getAt(i, j, k);
    REAL d100 = getAt(i + 1, j, k);
    REAL d010 = getAt(i, j + 1, k);
    REAL d110 = getAt(i + 1, j + 1, k);
    REAL d001 = getAt(i, j, k + 1);
    REAL d101 = getAt(i + 1, j, k + 1);
    REAL d011 = getAt(i, j + 1, k + 1);
    REAL d111 = getAt(i + 1, j + 1, k + 1);

    REAL dx00 = lerp(d000, d100, alpha);
    REAL dx10 = lerp(d010, d110, alpha);
    REAL dxy0 = lerp(dx00, dx10, beta);

    REAL dx01 = lerp(d001, d101, alpha);
    REAL dx11 = lerp(d011, d111, alpha);
    REAL dxy1 = lerp(dx01, dx11, beta);

    REAL d0y0 = lerp(d000, d010, beta);
    REAL d0y1 = lerp(d001, d011, beta);
    REAL d0yz = lerp(d0y0, d0y1, gamma);

    REAL d1y0 = lerp(d100, d110, beta);
    REAL d1y1 = lerp(d101, d111, beta);
    REAL d1yz = lerp(d1y0, d1y1, gamma);

    REAL dx0z = lerp(dx00, dx01, gamma);
    REAL dx1z = lerp(dx10, dx11, gamma);

    normal[0] = d0yz - d1yz;
    normal[1] = dx0z - dx1z;
    normal[2] = dxy0 - dxy1;

    REAL l = normal.length2();
    if (l < 0.0001f)
        normal = vec3f(0, 0, 0);
    else
        normal.normalize();

    d = (1.0f - gamma) * dxy0 + gamma * dxy1;
```

[常见几何体的SDF可参考](https://iquilezles.org/articles/distfunctions/)
