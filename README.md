# <span style="font-variant:small-caps;"><div align="center">CosmoPower</div></span>

## <div align="center">Neural network emulators for matter and CMB power spectra</div>

The content of this repository will be made publicly available upon acceptance of [our code release paper](https://arxiv.org/abs/2106.03846).
In the meantime, please do get in touch (_a dot spuriomancini at ucl dot ac dot uk_, or open an issue in this GitHub repository) if you are interested in using `CosmoPower` - we are happy to provide early access to the code!

---

`CosmoPower` is a suite of neural cosmological power spectrum emulators developed to accelerate by orders-of-magnitude parameter estimation from two-point statistics analyses of Large-Scale Structure (LSS) and Cosmic Microwave Background (CMB) surveys. The emulators replace the computation of matter and CMB power spectra from Boltzmann codes such as CAMB and CLASS, and they are tested against these codes for different survey configurations.

Please have a look [at our code release paper](https://arxiv.org/abs/2106.03846) for examples of application of `CosmoPower` to:

- [a 3x2pt analysis](https://doi.org/10.1093/mnras/sty551) from the KiDS and GAMA surveys

![alt text](https://github.com/alessiospuriomancini/cosmopower/blob/main/images/class_vs_cosmopower_kxg.png "KiDS-450 + GAMA")

---

- [a cosmic shear analysis](10.1051/0004-6361/202039070) from the KiDS survey

![alt text](https://github.com/alessiospuriomancini/cosmopower/blob/main/images/class_vs_cosmopower_k1k.png "KiDS-1000")

---

- a simulated _Euclid_-like cosmic shear analysis

![alt text](https://github.com/alessiospuriomancini/cosmopower/blob/main/images/class_vs_cosmopower_euclid.png "Euclid-like")

---

- [a _Planck_ 2018 TTTEEE analysis](http://dx.doi.org/10.1051/0004-6361/201833910)

![alt text](https://github.com/alessiospuriomancini/cosmopower/blob/main/images/class_vs_cosmopower_planck.png "Planck 2018")
