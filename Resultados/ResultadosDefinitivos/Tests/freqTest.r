library(dplyr)

ARI <- read.csv("ARI.csv", dec = ".", row.names = 1)
Unsat <- read.csv("Unsat.csv", dec = ".", row.names = 1)
Time <- read.csv("Time.csv", dec = ".", row.names = 1)

colnames(ARI) <- c("BRKGA", "SHADE")
colnames(Unsat) <- c("BRKGA", "SHADE")
colnames(Time) <- c("BRKGA", "SHADE")

#https://www.rdocumentation.org/packages/stats/versions/3.5.2/topics/wilcox.test
test.ari = wilcox.test(x = ARI$BRKGA, y = ARI$SHADE, paired = T, alternative = "l")
test.unsat = wilcox.test(Unsat$BRKGA, Unsat$SHADE, paired = T, alternative = "g")
test.time = wilcox.test(Time$BRKGA, Time$SHADE, paired = T, alternative = "g")

median(ARI$BRKGA)

#write.csv(ARI, "ARI.csv", row.names = T, col.names = T, sep = ",")
#write.csv(Unsat, "Unsat.csv", row.names = T, col.names = T, sep = ",")
#write.csv(Time, "Time.csv", row.names = T, col.names = T, sep = ",")
