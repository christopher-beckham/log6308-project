######################
# ITEM AUTOENCODER 2 #
######################

dfi2 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c2/results.txt")
dfi5 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c5/results.txt")
dfi10 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c10/results.txt")
dfi25 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c25/results.txt")
dfi50 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c50/results.txt")
dfi100 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c100/results.txt")
dfi200 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c200/results.txt")

plot(dfi2$train_i_loss,type="l", ylim=c(0,10))
lines(dfi5$train_i_loss,col="red")
lines(dfi10$train_i_loss,col="purple")
lines(dfi25$train_i_loss,col="green")
lines(dfi50$train_i_loss,col="blue")

plot(dfi2$valid_i_loss,type="l", ylim=c(0,10))
lines(dfi5$valid_i_loss,col="red")
lines(dfi10$valid_i_loss,col="purple")
lines(dfi25$valid_i_loss,col="green")
lines(dfi50$valid_i_loss,col="blue")

# -------------

dfi5 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c5/results.txt")
dfi10 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c10/results.txt")
dfi20 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c20/results.txt")
dfi30 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c30/results.txt")
dfi60 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c60/results.txt")

dfi50 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c50/results.txt")
dfi50.m5 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-5_c50/results.txt")
dfi50.m10 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-10_c50/results.txt")
dfi50.m20 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-20_c50/results.txt")
dfi50.m30 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-30_c50/results.txt")

par(mfrow=c(1,2))
for (tmode in c("train_i_loss", "valid_i_loss")) {
  cols=colorRampPalette(c("red", "orange"))(4)
  plot(dfi50[tmode][,1],type="l", xlab="epoch", ylab="MSE", main=tmode, xlim=c(0,25), lwd=3)
  lines(dfi50.m5[tmode],col=cols[1], lty="dotted",lwd=3)
  lines(dfi50.m10[tmode],col=cols[2], lty="dotted",lwd=3)
  lines(dfi50.m20[tmode],col=cols[3], lty="dotted",lwd=3)
  lines(dfi50.m30[tmode],col=cols[4], lty="dotted",lwd=3)
  lines(dfi30[tmode],col="darkgrey", lty="solid", lwd=3)
  capacity=c(6578483, 3647748, 3973663, 4625493, 5277323, 3973143)
  labs=c("k=50", "m=5", "m=10", "m=20", "m=30","k=30")
  legend("topright", legend=paste(labs, "(", capacity, ")",sep=""), col=c("black",cols),cex=0.6,lwd=rep(3,length(cols)),lty=rep("solid",length(cols)))
}


# -------------

dfi100 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_c100/results.txt")
dfi100.m10 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-10_c100/results.txt")
dfi100.m20 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-20_c100/results.txt")
dfi100.m40 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-40_c100/results.txt")
dfi100.m80 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-80_c100/results.txt")

par(mfrow=c(1,2))
for (tmode in c("train_i_loss", "valid_i_loss")) {
  cols=colorRampPalette(c("red", "orange"))(4)
  plot(dfi100[tmode][,1], type="l", xlab="epoch", ylab="MSE", main=tmode, lwd=3)
  lines(dfi100.m10[tmode],col=cols[1], lty="dotted",lwd=3)
  lines(dfi100.m20[tmode],col=cols[2], lty="dotted",lwd=3)
  lines(dfi100.m40[tmode],col=cols[3], lty="dotted",lwd=3)
  lines(dfi100.m80[tmode],col=cols[4], lty="dotted",lwd=3)
  lines(dfi60[tmode],col="grey", lty="solid", lwd=3)
  #lines(dfi25[tmode],col="grey", lty="solid", lwd=3)
  capacity=c(13091833, 7230863, 7883193, 9187853, 11797173, 6578483)
  labs=c("k=100", "m=10", "m=20", "m=40", "m=80","k=60")
  legend("topright", legend=paste(labs, "(", capacity, ")",sep=""), fill=c("black",cols,"grey"),cex=0.5)
}


####################
# USER AUTOENCODER #
####################

dfu30 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_c30/results.txt")

dfu50 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_c50/results.txt")
dfu50.m5 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_lowrank2_m1-5_c50/results.txt")
dfu50.m10 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_lowrank2_m1-10_c50/results.txt")
dfu50.m20 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_lowrank2_m1-20_c50/results.txt")
dfu50.m30 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-30_c50/results.txt")

par(mfrow=c(1,2))
for (tmode in c("train_u_loss", "valid_u_loss")) {
  cols=colorRampPalette(c("red", "orange"))(4)
  plot(dfu50[tmode][,1],type="l", xlab="epoch", ylab="MSE", main=tmode, xlim=c(0,25), lwd=3)
  lines(dfu50.m5[tmode],col=cols[1], lty="dotted",lwd=3)
  lines(dfu50.m10[tmode],col=cols[2], lty="dotted",lwd=3)
  lines(dfu50.m20[tmode],col=cols[3], lty="dotted",lwd=3)
  lines(dfu50.m30[tmode],col=cols[4], lty="dotted",lwd=3)
  lines(dfu30[tmode],col="darkgrey", lty="solid", lwd=3)
  capacity=c(7228317, 4008052, 4366137, 5082307, 5798477, 4365617) 
  labs=c("k=50", "k=50,m=5", "k=50,m=10", "k=50,m=20", "k=50,m=30","k=30")
  legend("topright", legend=paste(labs, "(", capacity, ")",sep=""), col=c("black",cols,"darkgrey"),cex=0.6,lwd=rep(3,length(cols)),lty=rep("solid",length(cols)))
}


# ---
par(mfrow=c(1,1))

dfu50.m5 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_lowrank2_m1-5_c50/results.txt")
dfu50.m5.1 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_lowrank2_m1-5_c50.1/results.txt")
dfi50.m5 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-5_c50/results.txt")
dfi50.m5.1 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-5_c50.1/results.txt")
dft50.m5 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_both_lowrank_tied_fixed_m5_c50/results.txt")
dft50.m5.1 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_both_lowrank_tied_fixed_m5_c50.1/results.txt")

plot(dfi50.m5$valid_i_loss, type="l",lwd=3)
lines(dfi50.m5.1$valid_i_loss, lwd=3)
lines(dft50.m5$valid_i_loss,col="red", lwd=3)
lines(dft50.m5.1$valid_i_loss,col="red", lwd=3)

plot(dfu50.m5$valid_u_loss, type="l", lwd=3)
lines(dfu50.m5.1$valid_u_loss, lwd=3)
lines(dft50.m5$valid_u_loss,col="red", lwd=3)
lines(dft50.m5.1$valid_u_loss,col="red", lwd=3)

dfu50.m10 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_lowrank2_m1-10_c50/results.txt")
dfu50.m10.1 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_lowrank2_m1-10_c50.1/results.txt")
dfi50.m10 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-10_c50/results.txt")
dfi50.m10.1 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-10_c50.1/results.txt")
dft50.m10 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_both_lowrank_tied_fixed_m10_c50/results.txt")
dft50.m10.1 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_both_lowrank_tied_fixed_m10_c50.1/results.txt")

plot(dfi50.m10$valid_i_loss, type="l")
lines(dfi50.m10.1$valid_i_loss)
lines(dft50.m10$valid_i_loss,col="red")
lines(dft50.m10.1$valid_i_loss,col="red")

plot(dfu50.m10$valid_u_loss, type="l")
lines(dfu50.m10.1$valid_u_loss)
lines(dft50.m10$valid_u_loss,col="red")
lines(dft50.m10.1$valid_u_loss,col="red")

dfu50.m20 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_lowrank2_m1-20_c50/results.txt")
dfu50.m20.1 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_lowrank2_m1-20_c50.1/results.txt")
dfi50.m20 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-20_c50/results.txt")
dfi50.m20.1 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-20_c50.1/results.txt")
dft50.m20 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_both_lowrank_tied_fixed_m20_c50/results.txt")
dft50.m20.1 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_both_lowrank_tied_fixed_m20_c50.1/results.txt")

plot(dfi50.m20$valid_i_loss, type="l")
lines(dfi50.m20.1$valid_i_loss)
lines(dft50.m20$valid_i_loss,col="red")
lines(dft50.m20.1$valid_i_loss,col="red")

plot(dfu50.m20$valid_u_loss, type="l")
lines(dfu50.m20.1$valid_u_loss)
lines(dft50.m20$valid_u_loss,col="red")
lines(dft50.m20.1$valid_u_loss,col="red")

dfu50.m30 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_lowrank2_m1-30_c50/results.txt")
dfu50.m30.1 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_user_lowrank2_m1-30_c50.1/results.txt")
dfi50.m30 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-30_c50/results.txt")
dfi50.m30.1 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-30_c50.1/results.txt")
dft50.m30 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_both_lowrank_tied_fixed_m30_c50.0/results.txt")
dft50.m30.1 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_both_lowrank_tied_fixed_m30_c50.1/results.txt")

plot(dfi50.m30$valid_i_loss, type="l")
lines(dfi50.m30.1$valid_i_loss)
lines(dft50.m30$valid_i_loss,col="red")
lines(dft50.m30.1$valid_i_loss,col="red")

plot(dfu50.m30$valid_u_loss, type="l")
lines(dfu50.m30.1$valid_u_loss)
lines(dft50.m30$valid_u_loss,col="red")
lines(dft50.m30.1$valid_u_loss,col="red")
