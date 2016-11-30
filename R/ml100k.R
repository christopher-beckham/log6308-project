dfu.50.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c50_lr0.1_bs1024/results.txt")
dfu.100.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c100_lr0.1_bs1024/results.txt")
dfu.200.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c200_lr0.1_bs1024/results.txt")
dfu.300.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c300_lr0.1_bs1024/results.txt")
dfu.600.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c600_lr0.1_bs1024/results.txt")
dfu.1000.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c1000_lr0.1_bs1024/results.txt")
dfu.5000.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c5000_lr0.1_bs1024/results.txt")

pdf("item_100k_vary_k.pdf",height=4)

par(mfrow=c(1,2))

plot(dfu.5000.100k$train_u_loss, type="l",lwd=2, xlab="epoch", ylab="train loss", main="item autoenc on ml-100k")
lines(dfu.1000.100k$train_u_loss,col="darkgrey",lwd=2)
lines(dfu.600.100k$train_u_loss,col="red",lwd=2)
lines(dfu.300.100k$train_u_loss,col="orange",lwd=2)
lines(dfu.200.100k$train_u_loss,col="darkgreen",lwd=2)
lines(dfu.100.100k$train_u_loss,col="blue",lwd=2)
lines(dfu.50.100k$train_u_loss,col="purple",lwd=2)
legend("topright", 
   legend=c(5000,1000,600,300,200,100,50), 
   col=c("black","darkgrey","red","orange","darkgreen","blue","purple"),
   lty="solid", lwd=2, cex=0.5
)

plot(dfu.5000.100k$valid_u_rmse, type="l",lwd=2, ylim=c(0.3,2.0), xlab="epoch", ylab="valid rmse", main="item autoenc on ml-100k")
lines(dfu.1000.100k$valid_u_rmse,col="darkgrey",lwd=2)
lines(dfu.600.100k$valid_u_rmse,col="red",lwd=2)
lines(dfu.300.100k$valid_u_rmse,col="orange",lwd=2)
lines(dfu.200.100k$valid_u_rmse,col="darkgreen",lwd=2)
lines(dfu.100.100k$valid_u_rmse,col="blue",lwd=2)
lines(dfu.50.100k$valid_u_rmse,col="purple",lwd=2)
legend("topright", 
       legend=c(5000,1000,600,300,200,100,50), 
       col=c("black","darkgrey","red","orange","darkgreen","blue","purple"),
       lty="solid", lwd=2,cex=0.5
)

dev.off()

# ---------

dfu.100.100k.untied = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_untied_c100_lr0.1_bs1024/results.txt")
dfu.200.100k.untied = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_untied_c200_lr0.1_bs1024/results.txt")
dfu.300.100k.untied = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_untied_c300_lr0.1_bs1024/results.txt")
dfu.600.100k.untied = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_untied_c600_lr0.1_bs1024/results.txt")
dfu.1000.100k.untied = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_untied_c1000_lr0.1_bs1024/results.txt")
dfu.5000.100k.untied = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_untied_c5000_lr0.1_bs1024/results.txt")

pdf("item_100k_tied_vs_untied.pdf", width=5, height=4)

par(mfrow=c(2,3))

plot(dfu.100.100k$valid_u_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=100")
lines(dfu.100.100k.untied$valid_u_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfu.200.100k$valid_u_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=200")
lines(dfu.200.100k.untied$valid_u_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfu.300.100k$valid_u_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=300")
lines(dfu.300.100k.untied$valid_u_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfu.600.100k$valid_u_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=600")
lines(dfu.600.100k.untied$valid_u_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfu.1000.100k$valid_u_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=1000")
lines(dfu.1000.100k.untied$valid_u_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfu.5000.100k$valid_u_rmse,type="l", ylim=c(0.3,1.2), lwd=2, xlab="epoch", ylab="valid rmse", main="k=5000")
lines(dfu.5000.100k.untied$valid_u_rmse,col="red", lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2, cex=0.5)

dev.off()

# ---

dfu.100.100k.sigma001 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c100_lr0.1_bs1024_sigma0.1/results.txt")
dfu.200.100k.sigma001 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c200_lr0.1_bs1024_sigma0.1/results.txt")
dfu.300.100k.sigma001 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c300_lr0.1_bs1024_sigma0.1/results.txt")
dfu.600.100k.sigma001 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c600_lr0.1_bs1024_sigma0.1/results.txt")
dfu.1000.100k.sigma001 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c1000_lr0.1_bs1024_sigma0.1/results.txt")
dfu.5000.100k.sigma001 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_sigm_tied_c5000_lr0.1_bs1024_sigma0.1/results.txt")

plot(dfu.100.100k$valid_u_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=100")
lines(dfu.100.100k.sigma001$valid_u_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("no sigma","sigma=0.01"),lty="solid",lwd=2,cex=0.5)

plot(dfu.200.100k$valid_u_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=200")
lines(dfu.200.100k.sigma001$valid_u_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfu.300.100k$valid_u_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=300")
lines(dfu.300.100k.sigma001$valid_u_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfu.600.100k$valid_u_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=600")
lines(dfu.600.100k.sigma001$valid_u_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfu.1000.100k$valid_u_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=1000")
lines(dfu.1000.100k.sigma001$valid_u_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfu.5000.100k$valid_u_rmse,type="l", ylim=c(0.3,1.2), lwd=2, xlab="epoch", ylab="valid rmse", main="k=5000")
lines(dfu.5000.100k.sigma001$valid_u_rmse,col="red", lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2, cex=0.5)











# --


dfi.50.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_tied_c50_lr0.1_bs1024/results.txt")
dfi.100.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_tied_c100_lr0.1_bs1024/results.txt")
dfi.200.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_tied_c200_lr0.1_bs1024/results.txt")
dfi.300.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_tied_c300_lr0.1_bs1024/results.txt")
dfi.600.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_tied_c600_lr0.1_bs1024/results.txt")
dfi.1000.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_tied_c1000_lr0.1_bs1024/results.txt")
dfi.5000.100k = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_tied_c5000_lr0.1_bs1024/results.txt")

pdf("user_100k_vary_k.pdf",height=4)

par(mfrow=c(1,2))

plot(dfi.5000.100k$train_i_loss, type="l",lwd=2, xlab="epoch", ylab="train loss", main="user autoenc on ml-100k")
lines(dfi.1000.100k$train_i_loss,col="darkgrey",lwd=2)
lines(dfi.600.100k$train_i_loss,col="red",lwd=2)
lines(dfi.300.100k$train_i_loss,col="orange",lwd=2)
lines(dfi.200.100k$train_i_loss,col="darkgreen",lwd=2)
lines(dfi.100.100k$train_i_loss,col="blue",lwd=2)
lines(dfi.50.100k$train_i_loss,col="purple",lwd=2)
legend("topright", 
       legend=c(5000,1000,600,300,200,100,50), 
       col=c("black","darkgrey","red","orange","darkgreen","blue","purple"),
       lty="solid", lwd=2, cex=0.5
)

plot(dfi.5000.100k$valid_i_rmse, type="l",lwd=2, xlab="epoch", ylab="train loss", main="user autoenc on ml-100k", ylim=c(0.4,2.0))
lines(dfi.1000.100k$valid_i_rmse,col="darkgrey",lwd=2)
lines(dfi.600.100k$valid_i_rmse,col="red",lwd=2)
lines(dfi.300.100k$valid_i_rmse,col="orange",lwd=2)
lines(dfi.200.100k$valid_i_rmse,col="darkgreen",lwd=2)
lines(dfi.100.100k$valid_i_rmse,col="blue",lwd=2)
lines(dfi.50.100k$valid_i_rmse,col="purple",lwd=2)
legend("topright", 
       legend=c(5000,1000,600,300,200,100,50), 
       col=c("black","darkgrey","red","orange","darkgreen","blue","purple"),
       lty="solid", lwd=2, cex=0.5
)

dev.off()

# -----

dfi.100.100k.untied = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_untied_c100_lr0.1_bs1024/results.txt")
dfi.200.100k.untied = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_untied_c200_lr0.1_bs1024/results.txt")
dfi.300.100k.untied = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_untied_c300_lr0.1_bs1024/results.txt")
dfi.600.100k.untied = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_untied_c600_lr0.1_bs1024/results.txt")
dfi.1000.100k.untied = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_untied_c1000_lr0.1_bs1024/results.txt")
dfi.5000.100k.untied = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_itemmask_sigm_untied_c5000_lr0.1_bs1024/results.txt")

pdf("user_100k_tied_vs_untied.pdf", width=5, height=4)

par(mfrow=c(2,3))

plot(dfi.100.100k$valid_i_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=100")
lines(dfi.100.100k.untied$valid_i_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfi.200.100k$valid_i_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=200")
lines(dfi.200.100k.untied$valid_i_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfi.300.100k$valid_i_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=300")
lines(dfi.300.100k.untied$valid_i_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfi.600.100k$valid_i_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=600")
lines(dfi.600.100k.untied$valid_i_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfi.1000.100k$valid_i_rmse,type="l",lwd=2, xlab="epoch", ylab="valid rmse", main="k=1000")
lines(dfi.1000.100k.untied$valid_i_rmse,col="red",lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2,cex=0.5)

plot(dfi.5000.100k$valid_i_rmse,type="l", ylim=c(0.3,1.2), lwd=2, xlab="epoch", ylab="valid rmse", main="k=5000")
lines(dfi.5000.100k.untied$valid_i_rmse,col="red", lwd=2)
legend("topright", col=c("black","red"),legend=c("tied","untied"),lty="solid",lwd=2, cex=0.5)

dev.off()



# ---

dfu.1000.100k.r0 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k_randomized//hybrid_usermask_sigm_untied_c1000_lr0.1_bs1024_rnd0/results.txt")
dfu.1000.100k.r1 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k_randomized//hybrid_usermask_sigm_untied_c1000_lr0.1_bs1024_rnd1/results.txt")
dfu.1000.100k.r2 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k_randomized//hybrid_usermask_sigm_untied_c1000_lr0.1_bs1024_rnd2/results.txt")
dfu.1000.100k.r3 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k_randomized//hybrid_usermask_sigm_untied_c1000_lr0.1_bs1024_rnd3/results.txt")

plot(dfu.1000.100k.r0$valid_u_rmse,type="l")
lines(dfu.1000.100k.r1$valid_u_rmse)
lines(dfu.1000.100k.r2$valid_u_rmse)
lines(dfu.1000.100k.r3$valid_u_rmse)

dfu.5000.100k.r0 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k_randomized//hybrid_usermask_sigm_untied_c5000_lr0.1_bs1024_rnd0/results.txt")
dfu.5000.100k.r1 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k_randomized//hybrid_usermask_sigm_untied_c5000_lr0.1_bs1024_rnd1/results.txt")
dfu.5000.100k.r2 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k_randomized//hybrid_usermask_sigm_untied_c5000_lr0.1_bs1024_rnd2/results.txt")
dfu.5000.100k.r3 = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k_randomized//hybrid_usermask_sigm_untied_c5000_lr0.1_bs1024_rnd3/results.txt")

plot(dfu.5000.100k.r0$valid_u_rmse,type="l")
lines(dfu.5000.100k.r1$valid_u_rmse)
lines(dfu.5000.100k.r2$valid_u_rmse)
lines(dfu.5000.100k.r3$valid_u_rmse)

# ---

pdf("item_relu_vs_tanh.pdf", width=5,height=4)

dfu.50.100k.tanh = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_tanh_tied_c50_lr0.1_bs1024/results.txt")
dfu.100.100k.tanh = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_tanh_tied_c100_lr0.1_bs1024/results.txt")
dfu.200.100k.tanh = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_tanh_tied_c200_lr0.1_bs1024/results.txt")
dfu.300.100k.tanh = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_tanh_tied_c300_lr0.1_bs1024/results.txt")
dfu.600.100k.tanh = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_tanh_tied_c600_lr0.1_bs1024/results.txt")
dfu.1000.100k.tanh = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_tanh_tied_c1000_lr0.1_bs1024/results.txt")
dfu.5000.100k.tanh = read.csv("~/Desktop/cuda4_4/github/log6308-project/output_100k/hybrid_usermask_tanh_tied_c5000_lr0.1_bs1024/results.txt")

par(mfrow=c(2,3))

plot(dfu.100.100k$valid_u_rmse,type="l", lwd=2, xlab="epoch", ylab="valid rmse", main="k=100")
lines(dfu.100.100k.tanh$valid_u_rmse,col="red", lwd=2)
legend("topright",legend=c("relu","tanh"),col=c("black","red"),lwd=2,cex=0.5,lty="solid")

plot(dfu.200.100k$valid_u_rmse,type="l", lwd=2, xlab="epoch", ylab="valid rmse",main="k=200")
lines(dfu.200.100k.tanh$valid_u_rmse,col="red", lwd=2)
legend("topright",legend=c("relu","tanh"),col=c("black","red"),lwd=2,cex=0.5,lty="solid")

plot(dfu.300.100k$valid_u_rmse,type="l", lwd=2, xlab="epoch", ylab="valid rmse",main="k=300")
lines(dfu.300.100k.tanh$valid_u_rmse,col="red", lwd=2)
legend("topright",legend=c("relu","tanh"),col=c("black","red"),lwd=2,cex=0.5,lty="solid")

plot(dfu.600.100k$valid_u_rmse,type="l", lwd=2, xlab="epoch", ylab="valid rmse", main="k=600")
lines(dfu.600.100k.tanh$valid_u_rmse,col="red", lwd=2)
legend("topright",legend=c("relu","tanh"),col=c("black","red"),lwd=2,cex=0.5,lty="solid")

plot(dfu.1000.100k$valid_u_rmse,type="l", lwd=2, xlab="epoch", ylab="valid rmse",main="k=1000")
lines(dfu.1000.100k.tanh$valid_u_rmse,col="red", lwd=2)
legend("topright",legend=c("relu","tanh"),col=c("black","red"),lwd=2,cex=0.5,lty="solid")

plot(dfu.5000.100k$valid_u_rmse,type="l", lwd=2, xlab="epoch", ylab="valid rmse",main="k=5000")
lines(dfu.5000.100k.tanh$valid_u_rmse,col="red", lwd=2)
legend("topright",legend=c("relu","tanh"),col=c("black","red"),lwd=2,cex=0.5,lty="solid")

dev.off()
