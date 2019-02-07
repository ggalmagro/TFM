
ARI <- read.csv("ARI.csv", dec = ",", row.names = 1)
Unsat <- read.csv("Unsat.csv", dec = ",", row.names = 1)
Time <- read.csv("Time.csv", dec = ",", row.names = 1)

colnames(ARI) <- c("BRKGA", "SHADE")
colnames(Unsat) <- c("BRKGA", "SHADE")
colnames(Time) <- c("BRKGA", "SHADE")

write.csv(ARI, "ARI.csv", row.names = T, col.names = T, sep = ",")
write.csv(Unsat, "Unsat.csv", row.names = T, col.names = T, sep = ",")
write.csv(Time, "Time.csv", row.names = T, col.names = T, sep = ",")
