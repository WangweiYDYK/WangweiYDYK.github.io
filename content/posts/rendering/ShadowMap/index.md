+++
title = 'Shadow'
date = 2025-02-22T13:38:09+08:00
draft = false
+++

# Shadow

## 光照贴图

## Shadow Map

> Shadow Map存在的问题：  
> Shadow Acne  
> Shadow Bias


```glsl
float ShadowCalculation(vec4 fragPosLightSpace)
{
    // 执行透视除法，将裁切空间坐标的范围-w到w转为-1到1。当使用正交投影矩阵，顶点w元素仍保持不变，所以这一步实际上毫无意义。可是，当使用透视投影的时候就是必须的了，所以为了保证在两种投影矩阵下都有效就得留着这行。
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // 变换到[0,1]的范围
    projCoords = projCoords * 0.5 + 0.5;
    // 取得最近点的深度(使用[0,1]范围下的fragPosLight当坐标)
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    // 取得当前片段在光源视角下的深度
    float currentDepth = projCoords.z;
    // 检查当前片段是否在阴影中
    float shadow = currentDepth > closestDepth  ? 1.0 : 0.0;

    return shadow;
}
```

### PCF

```glsl
float shadow = 0.0;
vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
for(int x = -1; x <= 1; ++x)
{
    for(int y = -1; y <= 1; ++y)
    {
        float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
        shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
    }    
}
shadow /= 9.0;
```


## 参考资料
[实时阴影技术（1）Shadow Mapping](https://www.cnblogs.com/KillerAery/p/15201310.html)