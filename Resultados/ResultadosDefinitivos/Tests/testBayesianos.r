library(rNPBST)

ARI <- read.csv("ARI.csv", dec = ".", row.names = 1)
Unsat <- read.csv("Unsat.csv", dec = ".", row.names = 1)
Time <- read.csv("Time.csv", dec = ".", row.names = 1)

#n.samples = paramatero para el numero de puntitos

ari.test = rNPBST::bayesianSign.test(ARI$BRKGA, ARI$SHADE)
unsat.test = rNPBST::bayesianSign.test(Unsat$BRKGA, Unsat$SHADE)
time.test = rNPBST::bayesianSign.test(Unsat$BRKGA, Unsat$SHADE)

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
