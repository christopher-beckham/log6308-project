


user.tmp600.tied.sigm2.bs1024 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new3/hybrid_usermask_sigm_tied_c600_lr0.1_bs1024/results.txt")
user.tmp600.l1.5.tied.sigm2.bs1024 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new3/hybrid_usermask_sigm_tied_c600_l1-1.000000e-05_lr0.1_bs1024/results.txt")
user.tmp600.l1.6.tied.sigm2.bs1024 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new3/hybrid_usermask_sigm_tied_c600_l1-1.000000e-06_lr0.1_bs1024/results.txt")

user.tmp300.tied.sigm2.bs1024 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new3/hybrid_usermask_sigm_tied_c300_lr0.1_bs1024/results.txt")
user.tmp1000.tied.sigm2.bs1024 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new3/hybrid_usermask_sigm_tied_c1000_lr0.1_bs1024/results.txt")

user.tmp500.500.tied.sigm2.bs1024 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new3/hybrid_usermask_sigm_tied_c500_c500_lr0.1_bs1024/results.txt")
user.tmp200x3.tied.sigm2.bs1024 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new3/hybrid_usermask_sigm_tied_c200,200,200_lr0.1_bs1024/results.txt")

pdf("item_10m_vary_k.pdf", height=4)

par(mfrow=c(1,2))

plot(user.tmp300.tied.sigm2.bs1024$train_u_loss,lwd=2,type="l", xlab="epoch", ylab="train loss")
lines(user.tmp600.tied.sigm2.bs1024$train_u_loss,lwd=2,col="orange")
lines(user.tmp1000.tied.sigm2.bs1024$train_u_loss,lwd=2,col="purple")
lines(user.tmp500.500.tied.sigm2.bs1024$train_u_loss,lwd=2,col="blue")
lines(user.tmp200x3.tied.sigm2.bs1024$train_u_loss,lwd=2,col="red")
legend("topright",
       legend=c("300","600","1000","500-500","200-200-200"),
       col=c("black","orange","purple","blue","red"),
       lty="solid",
       lwd=2,cex=0.5)

plot(user.tmp300.tied.sigm2.bs1024$valid_u_rmse,lwd=2,type="l", ylim=c(0.86, 0.90), xlim=c(0,80), xlab="epoch", ylab="valid rmse")
lines(user.tmp600.tied.sigm2.bs1024$valid_u_rmse,lwd=2,col="orange") # remove row # 24 from this
lines(user.tmp1000.tied.sigm2.bs1024$valid_u_rmse,lwd=2,col="purple") 
lines(user.tmp500.500.tied.sigm2.bs1024$valid_u_rmse,lwd=2,col="blue")
lines(user.tmp200x3.tied.sigm2.bs1024$valid_u_rmse,lwd=2,col="red")
legend("topright",
       legend=c("300","600","1000","500-500","200-200-200"),
       col=c("black","orange","purple","blue","red"),
       lty="solid",
       lwd=2,cex=0.5)

dev.off()

# ------

item.tmp600.tied.sigm2.bs1024 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new3/hybrid_itemmask_sigm_rp_tied_c600_lr0.1_bs1024/results.txt")
item.tmp1000.tied.sigm2.bs1024 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new3/hybrid_itemmask_sigm_rp_tied_c1000_lr0.1_bs1024/results.txt")

item.tmp1500.tied.sigm2.bs1024 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new3/hybrid_itemmask_sigm_rp_tied_c1500_lr0.1_bs1024/results.txt")
item.tmp2000.tied.sigm2.bs1024 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new3/hybrid_itemmask_sigm_rp_tied_c2000_lr0.1_bs1024/results.txt")

pdf("user_10m_vary_k.pdf", height=4)

par(mfrow=c(1,2))

plot(item.tmp600.tied.sigm2.bs1024$train_i_loss,type="l", lwd=2, ylim=c(0.00130, 0.00165), xlab="epoch", ylab="train loss")
lines(item.tmp1000.tied.sigm2.bs1024$train_i_loss,col="red", lwd=2)
lines(item.tmp1500.tied.sigm2.bs1024$train_i_loss,col="blue",lwd=2)
lines(item.tmp2000.tied.sigm2.bs1024$train_i_loss,col="orange",lwd=2)
legend("topright", legend=c("300","1000","1500","2000"), col=c("black","red","blue","orange"), lwd=2, lty="solid",cex=0.5)

plot(item.tmp600.tied.sigm2.bs1024$valid_i_rmse,type="l", ylim=c(0.87, 0.96), lwd=2, xlim=c(0,40), xlab="epoch", ylab="valid rmse")
lines(item.tmp1000.tied.sigm2.bs1024$valid_i_rmse,col="red", lwd=2)
lines(item.tmp1500.tied.sigm2.bs1024$valid_i_rmse,col="blue",lwd=2)
lines(item.tmp2000.tied.sigm2.bs1024$valid_i_rmse,col="orange",lwd=2)
legend("topright", legend=c("300","1000","1500","2000"), col=c("black","red","blue","orange"), lwd=2, lty="solid",cex=0.5)

dev.off()


