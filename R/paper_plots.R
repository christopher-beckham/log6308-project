
tmp = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-5_c50/results.txt")
tmp2 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-10_c50/results.txt")
tmp3 = read.csv("/Users/cjb60/Desktop/cuda4_4/github/log6308-project/output_new/hybrid_item_lowrank2_m1-20_c50/results.txt")

plot(dfi25$train_i_loss,type="l")
lines(tmp$train_i_loss,type="l",col="red")

plot(dfi50$valid_i_loss,type="l")
lines(tmp$valid_i_loss,type="l",col="red")
lines(tmp2$valid_i_loss,type="l",col="blue")
lines(tmp3$valid_i_loss,type="l",col="purple")

plot(dfu25$valid_u_loss,type="l",xlim=c(0,25))
lines(tmp$valid_u_loss,type="l",col="red")


####################
# ITEM AUTOENCODER #
####################

dfi25 = read.csv("../output_new/hybrid_item_c25/results.txt")
dfi25.mask = read.csv("../output_new/hybrid_item_mask_c25/results.txt")
dfi50 = read.csv("../output_new/hybrid_item_c50/results.txt")
dfi50.mask = read.csv("../output_new/hybrid_item_mask_c50/results.txt")
dfi100 = read.csv("../output_new/hybrid_item_c100/results.txt")
dfi100.mask = read.csv("../output_new/hybrid_item_mask_c100/results.txt")
dfi200 = read.csv("../output_new/hybrid_item_c200/results.txt")
dfi200.mask = read.csv("../output_new/hybrid_item_mask_c200/results.txt")

pdf("shallow_item.pdf",height=4)
par(mfrow=c(1,2))
for(col.name in c("train_i_loss", "valid_i_loss")) {
  plot(sqrt(dfi25[,col.name]), type="l", lwd=1, ylim=c(0.0,1), xlab="epoch", ylab="RMSE", main="One-layer AE \n(valid)")
  lines(sqrt(dfi25.mask[,col.name]),lwd=1, lty="dashed")
  lines(sqrt(dfi50[,col.name]), type="l", col="purple", lwd=1)
  lines(sqrt(dfi50.mask[,col.name]),lwd=1, col="purple", lty="dashed")
  lines(sqrt(dfi100[,col.name]), type="l", col="red", lwd=1)
  lines(sqrt(dfi100.mask[,col.name]),lwd=1, col="red", lty="dashed")
  lines(sqrt(dfi200[,col.name]), type="l", col="blue", lwd=1)
  lines(sqrt(dfi200.mask[,col.name]), type="l", col="blue", lwd=1, lty="dashed")
  legend("bottomright", 
         legend=c("25", "50","100","200","25mask","50mask","100mask","200mask"), 
         cex=0.5,
         lty=c("solid","solid","solid","solid","dashed","dashed","dashed","dashed"), 
         col=c("black", "purple", "red", "blue", "black", "purple", "red", "blue"))  
}
dev.off()

####################
# USER AUTOENCODER #
####################

dfu25 = read.csv("../output_new/hybrid_user_c25/results.txt")
dfu25.mask = read.csv("../output_new/hybrid_user_mask_c25/results.txt")
dfu50 = read.csv("../output_new/hybrid_user_c50/results.txt")
dfu50.mask = read.csv("../output_new/hybrid_user_mask_c50/results.txt")
dfu100 = read.csv("../output_new/hybrid_user_c100/results.txt")
dfu100.mask = read.csv("../output_new/hybrid_user_mask_c100/results.txt")
dfu200 = read.csv("../output_new/hybrid_user_c200/results.txt")
dfu200.mask = read.csv("../output_new/hybrid_user_mask_c200/results.txt")

pdf("shallow_user.pdf",height=4)
par(mfrow=c(1,2))
for(col.name in c("train_u_loss", "valid_u_loss")) {
  plot(dfu25[,col.name], type="l", lwd=1, ylim=c(0.3,1), xlab="epoch", ylab="MSE")
  lines(dfu25.mask[,col.name],lwd=1, lty="dashed")
  lines(dfu50[,col.name], type="l", col="purple", lwd=1)
  lines(dfu50.mask[,col.name],lwd=1, lty="dashed", col="purple")
  lines(dfu100[,col.name], type="l", col="red", lwd=1)
  lines(dfu100.mask[,col.name],lwd=1, lty="dashed", col="red")
  lines(dfu200[,col.name], type="l", col="blue", lwd=1)
  lines(dfu200.mask[,col.name],lwd=1, lty="dashed", col="blue")
  legend("bottomright", 
         legend=c("25","50","100","200","25mask", "50mask","100mask","200mask"), 
         lty=c("solid","solid","solid","solid","dashed","dashed","dashed","dashed"), 
         col=c("black", "purple", "red", "blue", "black", "purple", "red", "blue"),cex=0.5)
}
dev.off()


###

###

###########################
# DEEPER ITEM AUTOENCODER #
###########################

dfid25 = read.csv("../output_new/hybrid_item_deeper_c25/results.txt")
dfid50 = read.csv("../output_new/hybrid_item_deeper_c50/results.txt")
dfid100 = read.csv("../output_new/hybrid_item_deeper_c100/results.txt")
dfid200 = read.csv("../output_new/hybrid_item_deeper_c200/results.txt")

plot(dfid25$train_i_loss,type="l",lwd=2, ylim=c(0,1))
lines(dfid50$train_i_loss,type="l",lwd=2,col="red")
lines(dfid100$train_i_loss,type="l",lwd=2,col="green")
lines(dfid200$train_i_loss,type="l",lwd=2,col="purple")

plot(dfid25$valid_i_loss,type="l",lwd=2, ylim=c(0,1))
lines(dfid50$valid_i_loss,type="l",lwd=2,col="red")
lines(dfid100$valid_i_loss,type="l",lwd=2,col="green")
lines(dfid200$valid_i_loss,type="l",lwd=2,col="purple")


###########################
# DEEPER USER AUTOENCODER #
###########################

dfud25 = read.csv("../output_new/hybrid_user_deeper_c25/results.txt")
dfud50 = read.csv("../output_new/hybrid_user_deeper_c50/results.txt")
dfud100 = read.csv("../output_new/hybrid_user_deeper_c100/results.txt")
dfud200 = read.csv("../output_new/hybrid_user_deeper_c200/results.txt")

plot(dfud25$train_u_loss,type="l",lwd=2, ylim=c(0,1))
lines(dfud50$train_u_loss,type="l",lwd=2,col="red")
lines(dfud100$train_u_loss,type="l",lwd=2,col="green")
lines(dfud200$train_u_loss,type="l",lwd=2,col="purple")

plot(dfud25$valid_u_loss,type="l",lwd=2, ylim=c(0,1))
lines(dfud50$valid_u_loss,type="l",lwd=2,col="red")
lines(dfud100$valid_u_loss,type="l",lwd=2,col="green")
lines(dfud200$valid_u_loss,type="l",lwd=2,col="purple")

###########################
# COMPARE SHALLOW/DEEP    #
###########################

pdf("test.pdf", height=5)

par(mfrow=c(2,4))

plot(sqrt(dfi25$valid_i_loss),type="l", xlab="epoch", ylab="RMSE")
lines(sqrt(dfid25$valid_i_loss), col="red")
legend("topright", legend=c("item-25","item-25-25"), col=c("black", "red"), lty=c("solid", "solid"),cex=0.5)
plot(sqrt(dfu25$valid_u_loss),type="l", xlab="epoch", ylab="RMSE")
lines(sqrt(dfud25$valid_u_loss), col="red")
legend("topright", legend=c("user-25","user-25-25"), col=c("black", "red"), lty=c("solid", "solid"),cex=0.5)

plot(sqrt(dfi50$valid_i_loss),type="l", xlab="epoch", ylab="RMSE")
lines(sqrt(dfid50$valid_i_loss), col="red")
legend("topright", legend=c("item-50","item-50-50"), col=c("black", "red"), lty=c("solid", "solid"),cex=0.5)
plot(sqrt(dfu50$valid_u_loss),type="l", xlab="epoch", ylab="RMSE")
lines(sqrt(dfud50$valid_u_loss), col="red")
legend("topright", legend=c("user-50","user-50-50"), col=c("black", "red"), lty=c("solid", "solid"),cex=0.5)

plot(sqrt(dfi100$valid_i_loss),type="l", xlab="epoch", ylab="RMSE")
lines(sqrt(dfid100$valid_i_loss), col="red")
legend("topright", legend=c("item-100","item-100-100"), col=c("black", "red"), lty=c("solid", "solid"), cex=0.5)
plot(sqrt(dfu100$valid_u_loss),type="l", xlab="epoch", ylab="RMSE")
lines(sqrt(dfud100$valid_u_loss), col="red")
legend("topright", legend=c("user-100","user-100-100"), col=c("black", "red"), lty=c("solid", "solid"), cex=0.5)

plot(sqrt(dfi200$valid_i_loss),type="l", xlab="epoch", ylab="RMSE")
lines(sqrt(dfid200$valid_i_loss), col="red")
legend("topright", legend=c("item-200","item-200-200"), col=c("black", "red"), lty=c("solid", "solid"), cex=0.5)
plot(sqrt(dfu200$valid_u_loss),type="l", xlab="epoch", ylab="RMSE")
lines(sqrt(dfud200$valid_u_loss), col="red")
legend("topright", legend=c("user-200","user-200-200"), col=c("black", "red"), lty=c("solid", "solid"), cex=0.5)

dev.off()

###########################
# DEEPER TIED AUTOENCODER #
###########################

dftd25 = read.csv("../output_new/hybrid_both_deeper_tied_doubly_fixed_c25/results.txt")
dftd50 = read.csv("../output_new/hybrid_both_deeper_tied_c50/results.txt")
dftd100 = read.csv("../output_new/hybrid_both_deeper_tied_c100/results.txt")
dftd200 = read.csv("../output_new/hybrid_both_deeper_tied_c200/results.txt")

# train loss for user25
plot(sqrt(dftd25$train_u_loss),type="l",lwd=2, ylim=c(0,1))
lines(sqrt(dfud25$train_u_loss),col="red",lwd=2)
# valid loss for user25
plot(sqrt(dftd25$valid_u_loss),type="l",lwd=2)
lines(sqrt(dfud25$valid_u_loss),col="red",lwd=2)
# train loss for item25
plot(sqrt(dftd25$train_i_loss),type="l",lwd=2, ylim=c(0,1))
lines(sqrt(dfid25$train_i_loss),col="red",lwd=2)
# valid loss for item25
plot(sqrt(dftd25$valid_i_loss),type="l",lwd=2)
lines(sqrt(dfid25$valid_i_loss),col="red",lwd=2)


# train loss user
plot(sqrt(dftd50$train_u_loss),type="l",lwd=2, ylim=c(0,1))
lines(sqrt(dfud50$train_u_loss),col="red",lwd=2)
# valid loss user
plot(sqrt(dftd50$valid_u_loss),type="l",lwd=2, ylim=c(0,1))
lines(sqrt(dfud50$valid_u_loss),col="red",lwd=2)
# train loss item
plot(sqrt(dftd50$train_i_loss),type="l",lwd=2, ylim=c(0,1))
lines(sqrt(dfid50$train_i_loss),col="red",lwd=2)
# valid loss item
plot(sqrt(dftd50$valid_i_loss),type="l",lwd=2, ylim=c(0,1))
lines(sqrt(dfid50$valid_i_loss),col="red",lwd=2)


# train loss user
plot(dftd100$train_u_loss,type="l",lwd=2, ylim=c(0,1))
lines(dfud100$train_u_loss,col="red",lwd=2)
# valid loss user
plot(dftd100$valid_u_loss,type="l",lwd=2, ylim=c(0,1))
lines(dfud100$valid_u_loss,col="red",lwd=2)
# train loss item
plot(sqrt(dftd100$train_i_loss),type="l",lwd=2)
lines(sqrt(dfid100$train_i_loss),col="red",lwd=2)
# valid loss item
plot(sqrt(dftd100$valid_i_loss),type="l",lwd=2)
lines(sqrt(dfid100$valid_i_loss),col="red",lwd=2)


# -------

dftd25 = read.csv("../output_new/hybrid_both_deeper_tied_fixed_c25/results.txt")
dftd50 = read.csv("../output_new/hybrid_both_deeper_tied_c50/results.txt")
dftd100 = read.csv("../output_new/hybrid_both_deeper_tied_c100/results.txt")
dftd200 = read.csv("../output_new/hybrid_both_deeper_tied_c200/results.txt")

# ---



dfi50.deep = read.csv("../output_new/hybrid_item_deeper_c50/results.txt")
dfi100.deep = read.csv("../output_new/hybrid_item_deeper_c100/results.txt")
dfi200.deep = read.csv("../output_new/hybrid_item_deeper_c200/results.txt")

plot(dfi100.deep$valid_i_loss,type="l",lwd=2, ylim=c(0,1))
lines(dfi200.deep$valid_i_loss,lwd=2,col="blue")