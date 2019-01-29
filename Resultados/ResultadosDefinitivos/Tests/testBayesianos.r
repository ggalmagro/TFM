library(rNPBST)

installed <- rownames(installed.packages())
pkgs <- dir("~/R/x86_64-pc-linux-gnu-library/3.4")
new <- setdiff(pkgs, installed)
new
install.packages(new)

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("graph", version = "3.8")
