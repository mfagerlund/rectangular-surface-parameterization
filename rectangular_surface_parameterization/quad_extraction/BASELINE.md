# Baseline Quality Metrics (Pre-Quantization)

**Date:** January 2026
**Method:** `--scale` parameter (UV multiplication workaround)

## The Problem

The current approach multiplies UV coordinates by a scale factor before quad extraction.
This increases quad density but does NOT fix singularity placement.

## Benchmark Results

```
====================================================================================================
BENCHMARK COMPARISON TABLE (--scale workaround)
====================================================================================================
Mesh                  Scale   Quads  MinAng  MaxAng  AngRMS  AspMax  JacMin  Irreg
----------------------------------------------------------------------------------------------------
sphere320              10.0      94    37.3   160.4   18.81    2.05   0.335      5
sphere320              20.0     400    54.3   136.4    9.78    1.76   0.690      5
sphere320              30.0     908    62.2   129.0    8.11    1.62   0.777      2
torus                  10.0      87    42.3   158.4   19.34    1.81   0.368      5
torus                  20.0     392    49.5   135.8   10.72    1.50   0.697      2
torus                  30.0     886    48.5   143.8    8.34    1.97   0.590      4
pig                    10.0      81    12.2   143.1   19.19    1.93   0.211      0
pig                    20.0     371    15.3   166.0   15.83    3.63   0.243      7
pig                    30.0     888     4.0   175.2   12.82    6.93   0.070      6
====================================================================================================
```

### Column Legend

| Column | Ideal | Description |
|--------|-------|-------------|
| MinAng | 90° | Worst (smallest) corner angle |
| MaxAng | 90° | Worst (largest) corner angle |
| AngRMS | 0° | RMS deviation from 90° |
| AspMax | 1.0 | Worst aspect ratio |
| JacMin | 1.0 | Minimum scaled Jacobian (negative = inverted) |
| Irreg | min | Interior vertices with valence ≠ 4 |

## Analysis

### Sphere (genus 0, 2 singularities expected)

✓ Quality **improves** with scale:
- Min angle: 37° → 62° (+67%)
- Aspect ratio: 2.05 → 1.62 (-21%)
- Jacobian: 0.335 → 0.777 (+132%)

Simple topology, singularities are isolated. Scaling helps.

### Torus (genus 1, 0 singularities expected)

~ Quality **stable**:
- Min angle: 42° → 48° (slight improvement)
- Aspect ratio: 1.81 → 1.97 (slight degradation)

No singularities needed, so scaling has minimal effect.

### Pig (complex geometry, multiple singularities)

✗ Quality **DEGRADES** with scale:
- Min angle: 12° → **4°** (-67%)
- Aspect ratio: 1.93 → **6.93** (+259%)
- Jacobian: 0.211 → **0.070** (-67%)

**This is the smoking gun.** Complex geometry with improperly-placed singularities
produces increasingly degenerate quads at higher scales.

## Why This Happens

When singularities land at non-integer UV coordinates (e.g., `(2.347, 5.891)`):

1. Integer iso-lines miss the singularity
2. Quads near singularities become stretched/skewed
3. Higher scale = finer grid = more quads affected by the same misplacement
4. Result: worse quality, not better

## What Quantization Will Fix

Proper quantization snaps singularities to integer coordinates `(2, 6)`:

1. Integer iso-lines converge at singularities
2. Irregular vertices (valence 3 or 5) form properly
3. Surrounding quads are well-shaped
4. Quality improves uniformly with density

## Expected Improvements (Post-Quantization)

| Metric | Current (pig, scale 30) | Target |
|--------|-------------------------|--------|
| Min angle | 4° | > 45° |
| Max angle | 175° | < 135° |
| Aspect ratio | 6.93 | < 2.0 |
| Min Jacobian | 0.070 | > 0.5 |

## Reproduction

```bash
python scripts/benchmark_quad_quality.py Mesh/sphere320.obj Mesh/torus.obj Mesh/pig.obj --scales 10,20,30
```

---

*This baseline will be compared against post-quantization results.*
