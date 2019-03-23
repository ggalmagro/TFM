library(rNPBST)

ARI <- read.csv("ARI_TFG.csv", dec = ",", row.names = 1)

#n.samples = paramatero para el numero de puntitos

ari.copkm = rNPBST::bayesianSign.test(ARI$COPKM[26:100], ARI$SHADE[26:100], n.samples = 2000)
ari.lcvqe = rNPBST::bayesianSign.test(ARI$LCVQE[26:100], ARI$SHADE[26:100], n.samples = 2000)
ari.rdpm = rNPBST::bayesianSign.test(ARI$RDPM[26:100], ARI$SHADE[26:100], n.samples = 2000)
ari.tvclust = rNPBST::bayesianSign.test(ARI$TVClust[26:100], ARI$SHADE[26:100], n.samples = 2000)
ari.cekm = rNPBST::bayesianSign.test(ARI$CEKM[26:100], ARI$SHADE[26:100], n.samples = 2000)

ari.copkm$probabilities
ari.lcvqe$probabilities
ari.rdpm$probabilities
ari.tvclust$probabilities
ari.cekm$probabilities

plot(ari.copkm)
plot(ari.lcvqe)
plot(ari.rdpm)
plot(ari.tvclust)
plot(ari.cekm)

pdf(file = "AriCOPKM.pdf", width= 6, height = 4, useDingbats=F)
plot(ari.copkm)
dev.off()

pdf(file = "AriLCVQE.pdf", width= 6, height = 4, useDingbats=F)
plot(ari.lcvqe)
dev.off()

pdf(file = "AriRDPM.pdf", width= 6, height = 4, useDingbats=F)
plot(ari.rdpm)
dev.off()

pdf(file = "AriTVClust.pdf", width= 6, height = 4, useDingbats=F)
plot(ari.tvclust)
dev.off()

pdf(file = "AriCEKM.pdf", width= 6, height = 4, useDingbats=F)
plot(ari.cekm)
dev.off()
