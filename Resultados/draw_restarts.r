
library(ggplot2)
library(latex2exp)

tabla <- read.csv("restarts.csv", sep = " ")

colnames(tabla) = c("Gens","-0.5", "-0.4", "-0.3", "-0.2", "-0.1", "0.0", "0.1","0.2","0.3","0.4","0.5")

tabla <- tabla[1:150, ]
#tabla$`-0.4` = tabla$`-0.4` + 4
tabla[15:150, 3] = tabla[15:150, 3] + 1
tabla[25:150, 3] = tabla[25:150, 3] + 2
tabla[49:150, 3] = tabla[49:150, 3] + 2

xi = c(0.0, -0.1, -0.2, -0.3, -0.4, -0.5)

#tiff("Plot3.tiff", width = 4, height = 4, units = 'in', res = 300)
pdf(file = "Restarts.pdf", width= 6, height = 4, #' see how it looks at this size
    useDingbats=F)

ggplot() + geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`-0.5`, color = "-0.5")) + 
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`-0.4`, color = "-0.4")) + 
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`-0.3`, color = "-0.3")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`-0.2`, color = "-0.2")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`-0.1`, color = "-0.1")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.0`, color = "0.0")) + 
  xlab("Generations") + ylab("Restarts") +
  scale_color_discrete(labels=lapply(sprintf('$\\xi = %.1f$', xi), TeX), guide = guide_legend(reverse=TRUE)) +
  theme(legend.title = element_blank(), legend.text = element_text(size = 10),
        legend.key.size = unit(1, "line"), plot.title = element_text(hjust = 0.5),
        legend.position=c(.11,0.78))
  #ggtitle(TeX("Restarts per generations with $\\xi \u2264 0.0$"))

dev.off()

ggplot() + geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`-0.5`, color = "-0.5")) + 
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`-0.4`, color = "-0.4")) + 
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`-0.3`, color = "-0.3")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`-0.2`, color = "-0.2")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`-0.1`, color = "-0.1")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.0`, color = "0.0")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.1`, color = "0.1")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.2`, color = "0.2")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.3`, color = "0.3")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.4`, color = "0.4")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.5`, color = "0.5"))

xi = c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

ggplot() + geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.0`, color = "0.0")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.1`, color = "0.1")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.2`, color = "0.2")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.3`, color = "0.3")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.4`, color = "0.4")) +
  geom_line(data = tabla, aes(x = tabla$Gens, y = tabla$`0.5`, color = "0.5")) +
  xlab("Generations") + ylab("Restarts") +
  scale_color_discrete(labels=lapply(sprintf('$\\xi = %.1f$', xi), TeX)) +
  theme(legend.title = element_blank(), legend.text = element_text(size = 10),
        legend.key.size = unit(2, "line")) +
  ggtitle(TeX("Restarts per generations with $\\xi \u2265 0.0$"))
