dfu.50.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_usermask_sigm_tied_c50_lr0.1_bs1024/results.txt")
dfu.100.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_usermask_sigm_tied_c100_lr0.1_bs1024/results.txt")
dfu.200.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_usermask_sigm_tied_c200_lr0.1_bs1024/results.txt")
dfu.300.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_usermask_sigm_tied_c300_lr0.1_bs1024/results.txt")
dfu.600.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_usermask_sigm_tied_c600_lr0.1_bs1024/results.txt")
dfu.1000.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_usermask_sigm_tied_c1000_lr0.1_bs1024/results.txt")
dfu.5000.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_usermask_sigm_tied_c5000_lr0.1_bs1024/results.txt")

pdf("item_1m_vary_k.pdf", height=4)

par(mfrow=c(1,2))

plot(dfu.5000.1m$train_u_loss, type="l",lwd=2, ylab="train loss", xlab="epoch")
lines(dfu.1000.1m$train_u_loss,col="darkgrey",lwd=2)
lines(dfu.600.1m$train_u_loss,col="red",lwd=2)
lines(dfu.300.1m$train_u_loss,col="orange",lwd=2)
lines(dfu.200.1m$train_u_loss,col="darkgreen",lwd=2)
lines(dfu.100.1m$train_u_loss,col="blue",lwd=2)
legend("topright", 
       legend=c(5000,1000,600,300,200,100), 
       col=c("black","darkgrey","red","orange","darkgreen","blue"),
       lty="solid", lwd=2, cex=0.5
)

plot(dfu.5000.1m$valid_u_rmse, type="l",lwd=2, ylab="valid rmse", xlab="epoch")
lines(dfu.1000.1m$valid_u_rmse,col="darkgrey",lwd=2)
lines(dfu.600.1m$valid_u_rmse,col="red",lwd=2)
lines(dfu.300.1m$valid_u_rmse,col="orange",lwd=2)
lines(dfu.200.1m$valid_u_rmse,col="darkgreen",lwd=2)
lines(dfu.100.1m$valid_u_rmse,col="blue",lwd=2)
legend("topright", 
       legend=c(5000,1000,600,300,200,100), 
       col=c("black","darkgrey","red","orange","darkgreen","blue"),
       lty="solid", lwd=2, cex=0.5
)

dev.off()

# -----------

pdf("user_1m_vary_k.pdf", height=4)

par(mfrow=c(1,2))

dfi.100.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_itemmask_sigm_tied_c100_lr0.1_bs1024/results.txt")
dfi.200.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_itemmask_sigm_tied_c200_lr0.1_bs1024/results.txt")
dfi.300.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_itemmask_sigm_tied_c300_lr0.1_bs1024/results.txt")
dfi.600.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_itemmask_sigm_tied_c600_lr0.1_bs1024/results.txt")
dfi.1000.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_itemmask_sigm_tied_c1000_lr0.1_bs1024/results.txt")
dfi.5000.1m = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_1m/hybrid_itemmask_sigm_tied_c5000_lr0.1_bs1024/results.txt")

plot(dfi.5000.1m$train_i_loss, type="l",lwd=2, ylab="train loss", xlab="epoch")
lines(dfi.1000.1m$train_i_loss,col="darkgrey",lwd=2)
lines(dfi.600.1m$train_i_loss,col="red",lwd=2)
lines(dfi.300.1m$train_i_loss,col="orange",lwd=2)
lines(dfi.200.1m$train_i_loss,col="darkgreen",lwd=2)
lines(dfi.100.1m$train_i_loss,col="blue",lwd=2)
legend("topright", 
       legend=c(5000,1000,600,300,200,100), 
       col=c("black","darkgrey","red","orange","darkgreen","blue"),
       lty="solid", lwd=2, cex=0.5
)

plot(dfi.5000.1m$valid_i_rmse, type="l",lwd=2, ylab="valid rmse", xlab="epoch")
lines(dfi.1000.1m$valid_i_rmse,col="darkgrey",lwd=2)
lines(dfi.600.1m$valid_i_rmse,col="red",lwd=2)
lines(dfi.300.1m$valid_i_rmse,col="orange",lwd=2)
lines(dfi.200.1m$valid_i_rmse,col="darkgreen",lwd=2)
lines(dfi.100.1m$valid_i_rmse,col="blue",lwd=2)
legend("topright", 
       legend=c(5000,1000,600,300,200,100), 
       col=c("black","darkgrey","red","orange","darkgreen","blue"),
       lty="solid", lwd=2, cex=0.5
)

dev.off()

