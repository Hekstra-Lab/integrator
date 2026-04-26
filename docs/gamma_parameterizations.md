# Gamma Distribution Parameterizations

LaTeX code for the four gamma distribution parameterizations (A-D). Requires `\usepackage{listings}` and `\usepackage{amsmath}` in your preamble.

## Parameterization A

Direct parameterization of concentration and rate:

```
\begin{lstlisting}[caption={Parameterization A: Direct $(k, r)$}, label={lst:gamma-a}]
\begin{align}
k &= \text{softplus}(W_k \mathbf{x} + b_k) + \epsilon \\
r &= \text{softplus}(W_r \mathbf{x'} + b_r) + \epsilon \\
X &\sim \text{Gamma}(k, r)
\end{align}
\end{lstlisting}
```

## Parameterization B

Mean-Fano factor parameterization:

```
\begin{lstlisting}[caption={Parameterization B: Mean-Fano $(\mu, F)$}, label={lst:gamma-b}]
\begin{align}
\mu &= \text{softplus}(W_\mu \mathbf{x} + b_\mu) + \epsilon \\
F &= \text{softplus}(W_F \mathbf{x'} + b_F) + \epsilon \\
r &= \frac{1}{F + \epsilon} \\
k &= \mu \cdot r \\
X &\sim \text{Gamma}(k, r)
\end{align}
\end{lstlisting}
```

## Parameterization C

Mean-dispersion parameterization:

```
\begin{lstlisting}[caption={Parameterization C: Mean-Dispersion $(\mu, \phi)$}, label={lst:gamma-c}]
\begin{align}
\mu &= \text{softplus}(W_\mu \mathbf{x} + b_\mu) + \epsilon \\
\phi &= \text{softplus}(W_\phi \mathbf{x'} + b_\phi) + \epsilon \\
k &= \frac{1}{\phi + \epsilon} \\
r &= \frac{1}{\phi \mu + \epsilon} \\
X &\sim \text{Gamma}(k, r)
\end{align}
\end{lstlisting}
```

## Parameterization D

Concentration-Fano factor parameterization:

```
\begin{lstlisting}[caption={Parameterization D: Concentration-Fano $(k, F)$}, label={lst:gamma-d}]
\begin{align}
k &= \text{softplus}(W_k \mathbf{x} + b_k) + \epsilon \\
F &= \text{softplus}(W_F \mathbf{x'} + b_F) + \epsilon \\
r &= \frac{1}{F + \epsilon} \\
X &\sim \text{Gamma}(k, r)
\end{align}
\end{lstlisting}
```

## Summary Table

```
\begin{table}[h]
\centering
\begin{tabular}{lll}
\hline
Parameterization & Learned Parameters & Derived Parameters \\
\hline
A & $k$, $r$ & --- \\
B & $\mu$, $F$ (Fano) & $r = 1/F$, $k = \mu r$ \\
C & $\mu$, $\phi$ (dispersion) & $k = 1/\phi$, $r = 1/(\phi\mu)$ \\
D & $k$, $F$ (Fano) & $r = 1/F$ \\
\hline
\end{tabular}
\caption{Summary of gamma parameterizations}
\label{tab:gamma-params}
\end{table}
```

## Notes

- $\epsilon$ is a small constant (default: $10^{-6}$) for numerical stability
- $\mathbf{x}$ and $\mathbf{x'}$ are input feature vectors (may be the same or different)
- $\text{softplus}(z) = \log(1 + e^z)$ ensures positive outputs
