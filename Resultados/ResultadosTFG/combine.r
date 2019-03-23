
ARI <- read.csv("ARI_TFG.csv", dec = ".", row.names = 1)
write.csv(ARI[1:25, ], "ARI_TFG05.csv")
write.csv(ARI[26:50, ], "ARI_TFG10.csv")
write.csv(ARI[51:75, ], "ARI_TFG15.csv")
write.csv(ARI[76:100, ], "ARI_TFG20.csv")
