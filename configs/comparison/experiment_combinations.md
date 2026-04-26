# Experiment Combinations

All valid model configurations for the integrator framework.

## Constraints

- `cp_profile` requires group labels (hierarchical models only)
- `per_bin` and `wilson_per_bin` losses require bin-level priors (hierarchical models only)
- `qi` and `qbg` use the same surrogate type
- `fixed_basis_profile` and `per_bin_profile` excluded (require precomputed basis files)

## Component Summary

| Axis | Options | Count |
|------|---------|-------|
| Integrator | modela, modelb, modelc, hierarchicalA, hierarchicalB, hierarchicalC | 6 |
| Profile (qp) | dirichlet, learned_basis_profile, cp_profile | 3 |
| Intensity/Background (qi=qbg) | gammaA, gammaB, gammaC, gammaD, log_normal, folded_normal | 6 |
| Loss | default, per_bin, wilson_per_bin | 3 |

## All Valid Combinations (198)

| # | Integrator | qp | qi | qbg | Loss |
|--:|-----------|-----|-----|------|------|
| 1 | modela | dirichlet | gammaA | gammaA | default |
| 2 | modela | dirichlet | gammaB | gammaB | default |
| 3 | modela | dirichlet | gammaC | gammaC | default |
| 4 | modela | dirichlet | gammaD | gammaD | default |
| 5 | modela | dirichlet | log_normal | log_normal | default |
| 6 | modela | dirichlet | folded_normal | folded_normal | default |
| 7 | modela | learned_basis_profile | gammaA | gammaA | default |
| 8 | modela | learned_basis_profile | gammaB | gammaB | default |
| 9 | modela | learned_basis_profile | gammaC | gammaC | default |
| 10 | modela | learned_basis_profile | gammaD | gammaD | default |
| 11 | modela | learned_basis_profile | log_normal | log_normal | default |
| 12 | modela | learned_basis_profile | folded_normal | folded_normal | default |
| 13 | modelb | dirichlet | gammaA | gammaA | default |
| 14 | modelb | dirichlet | gammaB | gammaB | default |
| 15 | modelb | dirichlet | gammaC | gammaC | default |
| 16 | modelb | dirichlet | gammaD | gammaD | default |
| 17 | modelb | dirichlet | log_normal | log_normal | default |
| 18 | modelb | dirichlet | folded_normal | folded_normal | default |
| 19 | modelb | learned_basis_profile | gammaA | gammaA | default |
| 20 | modelb | learned_basis_profile | gammaB | gammaB | default |
| 21 | modelb | learned_basis_profile | gammaC | gammaC | default |
| 22 | modelb | learned_basis_profile | gammaD | gammaD | default |
| 23 | modelb | learned_basis_profile | log_normal | log_normal | default |
| 24 | modelb | learned_basis_profile | folded_normal | folded_normal | default |
| 25 | modelc | dirichlet | gammaA | gammaA | default |
| 26 | modelc | dirichlet | gammaB | gammaB | default |
| 27 | modelc | dirichlet | gammaC | gammaC | default |
| 28 | modelc | dirichlet | gammaD | gammaD | default |
| 29 | modelc | dirichlet | log_normal | log_normal | default |
| 30 | modelc | dirichlet | folded_normal | folded_normal | default |
| 31 | modelc | learned_basis_profile | gammaA | gammaA | default |
| 32 | modelc | learned_basis_profile | gammaB | gammaB | default |
| 33 | modelc | learned_basis_profile | gammaC | gammaC | default |
| 34 | modelc | learned_basis_profile | gammaD | gammaD | default |
| 35 | modelc | learned_basis_profile | log_normal | log_normal | default |
| 36 | modelc | learned_basis_profile | folded_normal | folded_normal | default |
| 37 | hierarchicalA | dirichlet | gammaA | gammaA | default |
| 38 | hierarchicalA | dirichlet | gammaA | gammaA | per_bin |
| 39 | hierarchicalA | dirichlet | gammaA | gammaA | wilson_per_bin |
| 40 | hierarchicalA | dirichlet | gammaB | gammaB | default |
| 41 | hierarchicalA | dirichlet | gammaB | gammaB | per_bin |
| 42 | hierarchicalA | dirichlet | gammaB | gammaB | wilson_per_bin |
| 43 | hierarchicalA | dirichlet | gammaC | gammaC | default |
| 44 | hierarchicalA | dirichlet | gammaC | gammaC | per_bin |
| 45 | hierarchicalA | dirichlet | gammaC | gammaC | wilson_per_bin |
| 46 | hierarchicalA | dirichlet | gammaD | gammaD | default |
| 47 | hierarchicalA | dirichlet | gammaD | gammaD | per_bin |
| 48 | hierarchicalA | dirichlet | gammaD | gammaD | wilson_per_bin |
| 49 | hierarchicalA | dirichlet | log_normal | log_normal | default |
| 50 | hierarchicalA | dirichlet | log_normal | log_normal | per_bin |
| 51 | hierarchicalA | dirichlet | log_normal | log_normal | wilson_per_bin |
| 52 | hierarchicalA | dirichlet | folded_normal | folded_normal | default |
| 53 | hierarchicalA | dirichlet | folded_normal | folded_normal | per_bin |
| 54 | hierarchicalA | dirichlet | folded_normal | folded_normal | wilson_per_bin |
| 55 | hierarchicalA | learned_basis_profile | gammaA | gammaA | default |
| 56 | hierarchicalA | learned_basis_profile | gammaA | gammaA | per_bin |
| 57 | hierarchicalA | learned_basis_profile | gammaA | gammaA | wilson_per_bin |
| 58 | hierarchicalA | learned_basis_profile | gammaB | gammaB | default |
| 59 | hierarchicalA | learned_basis_profile | gammaB | gammaB | per_bin |
| 60 | hierarchicalA | learned_basis_profile | gammaB | gammaB | wilson_per_bin |
| 61 | hierarchicalA | learned_basis_profile | gammaC | gammaC | default |
| 62 | hierarchicalA | learned_basis_profile | gammaC | gammaC | per_bin |
| 63 | hierarchicalA | learned_basis_profile | gammaC | gammaC | wilson_per_bin |
| 64 | hierarchicalA | learned_basis_profile | gammaD | gammaD | default |
| 65 | hierarchicalA | learned_basis_profile | gammaD | gammaD | per_bin |
| 66 | hierarchicalA | learned_basis_profile | gammaD | gammaD | wilson_per_bin |
| 67 | hierarchicalA | learned_basis_profile | log_normal | log_normal | default |
| 68 | hierarchicalA | learned_basis_profile | log_normal | log_normal | per_bin |
| 69 | hierarchicalA | learned_basis_profile | log_normal | log_normal | wilson_per_bin |
| 70 | hierarchicalA | learned_basis_profile | folded_normal | folded_normal | default |
| 71 | hierarchicalA | learned_basis_profile | folded_normal | folded_normal | per_bin |
| 72 | hierarchicalA | learned_basis_profile | folded_normal | folded_normal | wilson_per_bin |
| 73 | hierarchicalA | cp_profile | gammaA | gammaA | default |
| 74 | hierarchicalA | cp_profile | gammaA | gammaA | per_bin |
| 75 | hierarchicalA | cp_profile | gammaA | gammaA | wilson_per_bin |
| 76 | hierarchicalA | cp_profile | gammaB | gammaB | default |
| 77 | hierarchicalA | cp_profile | gammaB | gammaB | per_bin |
| 78 | hierarchicalA | cp_profile | gammaB | gammaB | wilson_per_bin |
| 79 | hierarchicalA | cp_profile | gammaC | gammaC | default |
| 80 | hierarchicalA | cp_profile | gammaC | gammaC | per_bin |
| 81 | hierarchicalA | cp_profile | gammaC | gammaC | wilson_per_bin |
| 82 | hierarchicalA | cp_profile | gammaD | gammaD | default |
| 83 | hierarchicalA | cp_profile | gammaD | gammaD | per_bin |
| 84 | hierarchicalA | cp_profile | gammaD | gammaD | wilson_per_bin |
| 85 | hierarchicalA | cp_profile | log_normal | log_normal | default |
| 86 | hierarchicalA | cp_profile | log_normal | log_normal | per_bin |
| 87 | hierarchicalA | cp_profile | log_normal | log_normal | wilson_per_bin |
| 88 | hierarchicalA | cp_profile | folded_normal | folded_normal | default |
| 89 | hierarchicalA | cp_profile | folded_normal | folded_normal | per_bin |
| 90 | hierarchicalA | cp_profile | folded_normal | folded_normal | wilson_per_bin |
| 91 | hierarchicalB | dirichlet | gammaA | gammaA | default |
| 92 | hierarchicalB | dirichlet | gammaA | gammaA | per_bin |
| 93 | hierarchicalB | dirichlet | gammaA | gammaA | wilson_per_bin |
| 94 | hierarchicalB | dirichlet | gammaB | gammaB | default |
| 95 | hierarchicalB | dirichlet | gammaB | gammaB | per_bin |
| 96 | hierarchicalB | dirichlet | gammaB | gammaB | wilson_per_bin |
| 97 | hierarchicalB | dirichlet | gammaC | gammaC | default |
| 98 | hierarchicalB | dirichlet | gammaC | gammaC | per_bin |
| 99 | hierarchicalB | dirichlet | gammaC | gammaC | wilson_per_bin |
| 100 | hierarchicalB | dirichlet | gammaD | gammaD | default |
| 101 | hierarchicalB | dirichlet | gammaD | gammaD | per_bin |
| 102 | hierarchicalB | dirichlet | gammaD | gammaD | wilson_per_bin |
| 103 | hierarchicalB | dirichlet | log_normal | log_normal | default |
| 104 | hierarchicalB | dirichlet | log_normal | log_normal | per_bin |
| 105 | hierarchicalB | dirichlet | log_normal | log_normal | wilson_per_bin |
| 106 | hierarchicalB | dirichlet | folded_normal | folded_normal | default |
| 107 | hierarchicalB | dirichlet | folded_normal | folded_normal | per_bin |
| 108 | hierarchicalB | dirichlet | folded_normal | folded_normal | wilson_per_bin |
| 109 | hierarchicalB | learned_basis_profile | gammaA | gammaA | default |
| 110 | hierarchicalB | learned_basis_profile | gammaA | gammaA | per_bin |
| 111 | hierarchicalB | learned_basis_profile | gammaA | gammaA | wilson_per_bin |
| 112 | hierarchicalB | learned_basis_profile | gammaB | gammaB | default |
| 113 | hierarchicalB | learned_basis_profile | gammaB | gammaB | per_bin |
| 114 | hierarchicalB | learned_basis_profile | gammaB | gammaB | wilson_per_bin |
| 115 | hierarchicalB | learned_basis_profile | gammaC | gammaC | default |
| 116 | hierarchicalB | learned_basis_profile | gammaC | gammaC | per_bin |
| 117 | hierarchicalB | learned_basis_profile | gammaC | gammaC | wilson_per_bin |
| 118 | hierarchicalB | learned_basis_profile | gammaD | gammaD | default |
| 119 | hierarchicalB | learned_basis_profile | gammaD | gammaD | per_bin |
| 120 | hierarchicalB | learned_basis_profile | gammaD | gammaD | wilson_per_bin |
| 121 | hierarchicalB | learned_basis_profile | log_normal | log_normal | default |
| 122 | hierarchicalB | learned_basis_profile | log_normal | log_normal | per_bin |
| 123 | hierarchicalB | learned_basis_profile | log_normal | log_normal | wilson_per_bin |
| 124 | hierarchicalB | learned_basis_profile | folded_normal | folded_normal | default |
| 125 | hierarchicalB | learned_basis_profile | folded_normal | folded_normal | per_bin |
| 126 | hierarchicalB | learned_basis_profile | folded_normal | folded_normal | wilson_per_bin |
| 127 | hierarchicalB | cp_profile | gammaA | gammaA | default |
| 128 | hierarchicalB | cp_profile | gammaA | gammaA | per_bin |
| 129 | hierarchicalB | cp_profile | gammaA | gammaA | wilson_per_bin |
| 130 | hierarchicalB | cp_profile | gammaB | gammaB | default |
| 131 | hierarchicalB | cp_profile | gammaB | gammaB | per_bin |
| 132 | hierarchicalB | cp_profile | gammaB | gammaB | wilson_per_bin |
| 133 | hierarchicalB | cp_profile | gammaC | gammaC | default |
| 134 | hierarchicalB | cp_profile | gammaC | gammaC | per_bin |
| 135 | hierarchicalB | cp_profile | gammaC | gammaC | wilson_per_bin |
| 136 | hierarchicalB | cp_profile | gammaD | gammaD | default |
| 137 | hierarchicalB | cp_profile | gammaD | gammaD | per_bin |
| 138 | hierarchicalB | cp_profile | gammaD | gammaD | wilson_per_bin |
| 139 | hierarchicalB | cp_profile | log_normal | log_normal | default |
| 140 | hierarchicalB | cp_profile | log_normal | log_normal | per_bin |
| 141 | hierarchicalB | cp_profile | log_normal | log_normal | wilson_per_bin |
| 142 | hierarchicalB | cp_profile | folded_normal | folded_normal | default |
| 143 | hierarchicalB | cp_profile | folded_normal | folded_normal | per_bin |
| 144 | hierarchicalB | cp_profile | folded_normal | folded_normal | wilson_per_bin |
| 145 | hierarchicalC | dirichlet | gammaA | gammaA | default |
| 146 | hierarchicalC | dirichlet | gammaA | gammaA | per_bin |
| 147 | hierarchicalC | dirichlet | gammaA | gammaA | wilson_per_bin |
| 148 | hierarchicalC | dirichlet | gammaB | gammaB | default |
| 149 | hierarchicalC | dirichlet | gammaB | gammaB | per_bin |
| 150 | hierarchicalC | dirichlet | gammaB | gammaB | wilson_per_bin |
| 151 | hierarchicalC | dirichlet | gammaC | gammaC | default |
| 152 | hierarchicalC | dirichlet | gammaC | gammaC | per_bin |
| 153 | hierarchicalC | dirichlet | gammaC | gammaC | wilson_per_bin |
| 154 | hierarchicalC | dirichlet | gammaD | gammaD | default |
| 155 | hierarchicalC | dirichlet | gammaD | gammaD | per_bin |
| 156 | hierarchicalC | dirichlet | gammaD | gammaD | wilson_per_bin |
| 157 | hierarchicalC | dirichlet | log_normal | log_normal | default |
| 158 | hierarchicalC | dirichlet | log_normal | log_normal | per_bin |
| 159 | hierarchicalC | dirichlet | log_normal | log_normal | wilson_per_bin |
| 160 | hierarchicalC | dirichlet | folded_normal | folded_normal | default |
| 161 | hierarchicalC | dirichlet | folded_normal | folded_normal | per_bin |
| 162 | hierarchicalC | dirichlet | folded_normal | folded_normal | wilson_per_bin |
| 163 | hierarchicalC | learned_basis_profile | gammaA | gammaA | default |
| 164 | hierarchicalC | learned_basis_profile | gammaA | gammaA | per_bin |
| 165 | hierarchicalC | learned_basis_profile | gammaA | gammaA | wilson_per_bin |
| 166 | hierarchicalC | learned_basis_profile | gammaB | gammaB | default |
| 167 | hierarchicalC | learned_basis_profile | gammaB | gammaB | per_bin |
| 168 | hierarchicalC | learned_basis_profile | gammaB | gammaB | wilson_per_bin |
| 169 | hierarchicalC | learned_basis_profile | gammaC | gammaC | default |
| 170 | hierarchicalC | learned_basis_profile | gammaC | gammaC | per_bin |
| 171 | hierarchicalC | learned_basis_profile | gammaC | gammaC | wilson_per_bin |
| 172 | hierarchicalC | learned_basis_profile | gammaD | gammaD | default |
| 173 | hierarchicalC | learned_basis_profile | gammaD | gammaD | per_bin |
| 174 | hierarchicalC | learned_basis_profile | gammaD | gammaD | wilson_per_bin |
| 175 | hierarchicalC | learned_basis_profile | log_normal | log_normal | default |
| 176 | hierarchicalC | learned_basis_profile | log_normal | log_normal | per_bin |
| 177 | hierarchicalC | learned_basis_profile | log_normal | log_normal | wilson_per_bin |
| 178 | hierarchicalC | learned_basis_profile | folded_normal | folded_normal | default |
| 179 | hierarchicalC | learned_basis_profile | folded_normal | folded_normal | per_bin |
| 180 | hierarchicalC | learned_basis_profile | folded_normal | folded_normal | wilson_per_bin |
| 181 | hierarchicalC | cp_profile | gammaA | gammaA | default |
| 182 | hierarchicalC | cp_profile | gammaA | gammaA | per_bin |
| 183 | hierarchicalC | cp_profile | gammaA | gammaA | wilson_per_bin |
| 184 | hierarchicalC | cp_profile | gammaB | gammaB | default |
| 185 | hierarchicalC | cp_profile | gammaB | gammaB | per_bin |
| 186 | hierarchicalC | cp_profile | gammaB | gammaB | wilson_per_bin |
| 187 | hierarchicalC | cp_profile | gammaC | gammaC | default |
| 188 | hierarchicalC | cp_profile | gammaC | gammaC | per_bin |
| 189 | hierarchicalC | cp_profile | gammaC | gammaC | wilson_per_bin |
| 190 | hierarchicalC | cp_profile | gammaD | gammaD | default |
| 191 | hierarchicalC | cp_profile | gammaD | gammaD | per_bin |
| 192 | hierarchicalC | cp_profile | gammaD | gammaD | wilson_per_bin |
| 193 | hierarchicalC | cp_profile | log_normal | log_normal | default |
| 194 | hierarchicalC | cp_profile | log_normal | log_normal | per_bin |
| 195 | hierarchicalC | cp_profile | log_normal | log_normal | wilson_per_bin |
| 196 | hierarchicalC | cp_profile | folded_normal | folded_normal | default |
| 197 | hierarchicalC | cp_profile | folded_normal | folded_normal | per_bin |
| 198 | hierarchicalC | cp_profile | folded_normal | folded_normal | wilson_per_bin |
