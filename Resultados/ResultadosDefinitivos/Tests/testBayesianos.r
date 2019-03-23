library(rNPBST)

ARI <- read.csv("ARI.csv", dec = ",", row.names = 1)
Unsat <- read.csv("Unsat.csv", dec = ",", row.names = 1)
Time <- read.csv("Time.csv", dec = ",", row.names = 1)

#n.samples = paramatero para el numero de puntitos

ari.test = rNPBST::bayesianSign.test(ARI$BRKGA[25:100], ARI$SHADE[25:100], n.samples = 2000)
unsat.test = rNPBST::bayesianSign.test(Unsat$BRKGA[25:100], Unsat$SHAD[25:100], n.samples = 2000)
time.test = rNPBST::bayesianSign.test(Unsat$BRKGA[25:100], Unsat$SHADE[25:100], n.samples = 2000)

ari.test$probabilities
unsat.test$probabilities
time.test$probabilities

pdf(file = "BayesSignoARI.pdf", width= 6, height = 4, useDingbats=F)
plot(ari.test)
dev.off()

pdf(file = "BayesSignoUnsat.pdf", width= 6, height = 4, useDingbats=F)
plot(unsat.test)
dev.off()

pdf(file = "BayesSignoTime.pdf", width= 6, height = 4, useDingbats=F)
plot(time.test)
dev.off()

ari.test = rNPBST::bayesianSignedRank.test(ARI$BRKGA, ARI$SHADE)
unsat.test = rNPBST::bayesianSignedRank.test(Unsat$SHADE, Unsat$BRKGA)
time.test = rNPBST::bayesianSignedRank.test(Unsat$SHADE, Unsat$BRKGA)

plot(ari.test)
plot(unsat.test)
plot(time.test)

ari.test = rNPBST::bayesian.imprecise(ARI$BRKGA, ARI$SHADE)
unsat.test = rNPBST::bayesian.imprecise(Unsat$SHADE, Unsat$BRKGA)
time.test = rNPBST::bayesian.imprecise(Unsat$SHADE, Unsat$BRKGA)

plot(ari.test)
plot(unsat.test)
plot(time.test)


ari.copkm = rNPBST::bayesianSign.test(ARI$COPKM[76:100], ARI$SHADE[76:100], n.samples = 2000)
ari.lcvqe = rNPBST::bayesianSign.test(ARI$LCVQE[76:100], ARI$SHADE[76:100], n.samples = 2000)
ari.rdpm = rNPBST::bayesianSign.test(ARI$RDPM[76:100], ARI$SHADE[76:100], n.samples = 2000)
ari.tvclust = rNPBST::bayesianSign.test(ARI$TVClust[76:100], ARI$SHADE[76:100], n.samples = 2000)
ari.cekm = rNPBST::bayesianSign.test(ARI$CEKM[76:100], ARI$SHADE[76:100], n.samples = 2000)

pdf(file = "AriCOPKM_20.pdf", width= 6, height = 4, useDingbats=F)
plot(ari.copkm)
dev.off()

pdf(file = "AriLCVQE_20.pdf", width= 6, height = 4, useDingbats=F)
plot(ari.lcvqe)
dev.off()

pdf(file = "AriRDPM_20.pdf", width= 6, height = 4, useDingbats=F)
plot(ari.rdpm)
dev.off()

pdf(file = "AriTVClust_20.pdf", width= 6, height = 4, useDingbats=F)
plot(ari.tvclust)
dev.off()

pdf(file = "AriCEKM_20.pdf", width= 6, height = 4, useDingbats=F)
plot(ari.cekm)
dev.off()
