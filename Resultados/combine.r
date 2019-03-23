
tabla5.1 <- read.csv("SHADEvsBRKGA/tabla5.csv", dec = ",", row.names = 1)
tabla5.2 <- read.csv("SHADEvsBRKGA-2/tabla5.csv", dec = ",", row.names = 1)

tabla10.1 <- read.csv("SHADEvsBRKGA/tabla10.csv", dec = ",", row.names = 1)
tabla10.2 <- read.csv("SHADEvsBRKGA-2/tabla10.csv", dec = ",", row.names = 1)

tabla15.1 <- read.csv("SHADEvsBRKGA/tabla15.csv", dec = ",", row.names = 1)
tabla15.2 <- read.csv("SHADEvsBRKGA-2/tabla15.csv", dec = ",", row.names = 1)

tabla20.1 <- read.csv("SHADEvsBRKGA/tabla20.csv", dec = ",", row.names = 1)
tabla20.2 <- read.csv("SHADEvsBRKGA-2/tabla20.csv", dec = ",", row.names = 1)

tabla5 = tabla5.1 * 0.4 + tabla5.2 * 0.6
tabla10 = tabla10.1 * 0.4 + tabla10.2 * 0.6
tabla15 = tabla15.1 * 0.4 + tabla15.2 * 0.6
tabla20 = tabla20.1 * 0.4 + tabla20.2 * 0.6

tabla5 = tabla5[!(row.names(tabla5) %in% 
                     c("Wine ", "Balance ", "Boston ", "Diabetes ", "Newthyroid ", "Heart ", "Rand ")), ]

tabla10 = tabla10[!(row.names(tabla10) %in% 
                      c("Wine ", "Balance ", "Boston ", "Diabetes ", "Newthyroid ", "Heart ", "Rand ")), ]

tabla15 = tabla15[!(row.names(tabla15) %in% 
                      c("Wine ", "Balance ", "Boston ", "Diabetes ", "Newthyroid ", "Heart ", "Rand ")), ]

tabla20 = tabla20[!(row.names(tabla20) %in% 
                      c("Wine ", "Balance ", "Boston ", "Diabetes ", "Newthyroid ", "Heart ", "Rand ")), ]

write.table(tabla5, "tabla5.dat", row.names = F, col.names = F)
write.table(tabla10, "tabla10.dat", row.names = F, col.names = F)
write.table(tabla15, "tabla15.dat", row.names = F, col.names = F)
write.table(tabla20, "tabla20.dat", row.names = F, col.names = F)
