
df1 = read.csv("~/Desktop/cudahead/github/log6308-project/output/simple_net_c50/results_train_raw.txt",header=F)
df2 = read.csv("~/Desktop/cudahead/github/log6308-project/output/simple_net_c100/results_train_raw.txt",header=F)
df3 = read.csv("~/Desktop/cudahead/github/log6308-project/output/simple_net_c200/results_train_raw.txt",header=F)
df4 = read.csv("~/Desktop/cudahead/github/log6308-project/output/simple_net_c300/results_train_raw.txt",header=F)
df5 = read.csv("~/Desktop/cudahead/github/log6308-project/output/simple_net_c400/results_train_raw.txt",header=F)

# ------

df1 = read.csv("~/Desktop/cudahead/github/log6308-project/output/simple_net_c50/results_valid_raw.txt",header=F)
df2 = read.csv("~/Desktop/cudahead/github/log6308-project/output/simple_net_c100/results_valid_raw.txt",header=F)
df3 = read.csv("~/Desktop/cudahead/github/log6308-project/output/simple_net_c200/results_valid_raw.txt",header=F)
df4 = read.csv("~/Desktop/cudahead/github/log6308-project/output/simple_net_c300/results_valid_raw.txt",header=F)
df5 = read.csv("~/Desktop/cudahead/github/log6308-project/output/simple_net_c400/results_valid_raw.txt",header=F)

plot(df1$V1,type="l")
lines(df2$V1,col="red")
lines(df3$V1,col="blue")
lines(df4$V1,col="green")
lines(df5$V1,col="purple")

# --


dft = read.csv("~/Desktop/cudahead/github/log6308-project/output/i_encoder_c500_c500_lr0.01/results.txt")


# ---------------
# 200 vs mask 200
# ---------------

mode="train_raw"
#mode="valid_raw"

df200 = read.csv(paste("~/Desktop/cudahead/github/log6308-project/output/simple_net_c200/results_",mode,".txt",sep=""),header=F)
df200.mask = read.csv(paste("~/Desktop/cudahead/github/log6308-project/output/simple_net_c200_mask/results_",mode,".txt",sep=""),header=F)

plot(df200$V1,type="l")
lines(df200.mask$V1,col="red")

# ---------------
# all vs mask all
# ---------------
#mode="train_raw"
mode="valid_raw"

par(mfrow=c(3,2))

for(bottleneck in c(50,100,200,300,400,500)) {
  df = read.csv(paste("~/Desktop/cudahead/github/log6308-project/output/simple_net_c",bottleneck,"/results_",mode,".txt",sep=""),header=F)
  df.mask = read.csv(paste("~/Desktop/cudahead/github/log6308-project/output/simple_net_c",bottleneck,"_mask/results_",mode,".txt",sep=""),header=F)
  plot(df$V1,type="l")
  lines(df.mask$V1,col="red")
  legend("topright", col=c("black","red"), lwd=c(1,1), lty=c("solid","solid"), legend=c("normal", "mask"))
}


# --------------
# 200 vs 200-200
# --------------

par(mfrow=c(1,1))

df200x2 = read.csv("~/Desktop/cudahead/github/log6308-project/output/i_encoder_c200_c200_lr0.01/results_train_raw.txt",header=F)
df200x2.adam = read.csv("~/Desktop/cudahead/github/log6308-project/output/i_encoder_c200_c200_lr0.01_adam/results_train_raw.txt",header=F)
df200 = read.csv("~/Desktop/cudahead/github/log6308-project/output/simple_net_c200/results_train_raw.txt",header=F)
plot(df200x2$V1,type="l",col="red")
lines(df200x2.adam$V1,col="green")
lines(df200$V1,col="black")


# ------------
# user encoder
# ------------

#mode="train"
mode="valid"

df1 = read.csv(paste("~/Desktop/cudahead/github/log6308-project/output/u_simple_net_c50/results_",mode,"_raw.txt",sep=""),header=F)
df2 = read.csv(paste("~/Desktop/cudahead/github/log6308-project/output/u_simple_net_c100/results_",mode,"_raw.txt",sep=""),header=F)
df3 = read.csv(paste("~/Desktop/cudahead/github/log6308-project/output/u_simple_net_c200/results_",mode,"_raw.txt",sep=""),header=F)
df4 = read.csv(paste("~/Desktop/cudahead/github/log6308-project/output/u_simple_net_c300/results_",mode,"_raw.txt",sep=""),header=F)
df5 = read.csv(paste("~/Desktop/cudahead/github/log6308-project/output/u_simple_net_c400/results_",mode,"_raw.txt",sep=""),header=F)

plot(df1$V1,type="l")
lines(df2$V1,col="red")
lines(df3$V1,col="blue")
lines(df4$V1,col="green")
lines(df5$V1,col="purple")

