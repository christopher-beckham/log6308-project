
####################
# ITEM AUTOENCODER #
####################

dfi50 = read.csv("../output_new/hybrid_item_c50/results.txt")
dfi50.mask = read.csv("../output_new/hybrid_item_mask_c50/results.txt")
dfi100 = read.csv("../output_new/hybrid_item_c100/results.txt")
dfi100.mask = read.csv("../output_new/hybrid_item_mask_c100/results.txt")
dfi200 = read.csv("../output_new/hybrid_item_c200/results.txt")
dfi200.mask = read.csv("../output_new/hybrid_item_mask_c200/results.txt")

plot(dfi50$train_i_loss, type="l", lwd=2, ylim=c(0,1), xlab="epoch", ylab="MSE")
lines(dfi50.mask$train_i_loss,lwd=2, lty="dashed")
lines(dfi100$train_i_loss, type="l", col="red", lwd=2)
lines(dfi100.mask$train_i_loss,lwd=2, col="red", lty="dashed")
lines(dfi200$train_i_loss, type="l", col="blue", lwd=2)
lines(dfi200.mask$train_i_loss, type="l", col="blue", lwd=2, lty="dashed")
legend("topright", 
       legend=c("50","100","200","50mask","100mask","200mask"), 
       lwd=c(2,2,2,2,2,2),
       lty=c("solid","solid","solid","dashed","dashed","dashed"), 
       col=c("black", "red", "blue", "black", "red", "blue"))

plot(dfi50$valid_i_loss, type="l", lwd=2, ylim=c(0.3,1), xlab="epoch", ylab="MSE")
lines(dfi50.mask$valid_i_loss,lwd=2, lty="dashed")
lines(dfi100$valid_i_loss, type="l", col="red", lwd=2)
lines(dfi100.mask$valid_i_loss, type="l", col="red", lwd=2, lty="dashed")
lines(dfi200$valid_i_loss, type="l", col="blue", lwd=2)
lines(dfi200.mask$valid_i_loss, type="l", col="blue", lwd=2, lty="dashed")
legend("topright", legend=c("50","100","200"), lwd=c(2,2,2), lty=c("solid","solid","solid"), col=c("black", "red","blue"))


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

plot(dfu25$train_u_loss, type="l", lwd=2, ylim=c(0.3,1), xlab="epoch", ylab="MSE")
lines(dfu25.mask$train_u_loss,lwd=2, lty="dashed")
lines(dfu50$train_u_loss, type="l", col="red", lwd=2)
lines(dfu50.mask$train_u_loss,lwd=2, lty="dashed", col="red")
lines(dfu100$train_u_loss, type="l", col="blue", lwd=2)
lines(dfu100.mask$train_u_loss,lwd=2, lty="dashed", col="blue")

legend("topright", 
       legend=c("50","100","200","50mask","100mask","200mask"), 
       lwd=c(2,2,2,2,2,2),
       lty=c("solid","solid","solid","dashed","dashed","dashed"), 
       col=c("black", "red", "blue", "black", "red", "blue"))

plot(dfu25$valid_u_loss, type="l", lwd=2, ylim=c(0.3,1), xlab="epoch", ylab="MSE")
lines(dfu25.mask$valid_u_loss,lwd=2, lty="dashed")
lines(dfu50$valid_u_loss, type="l", col="red", lwd=2)
lines(dfu50.mask$valid_u_loss,lwd=2, lty="dashed", col="red")
lines(dfu100$valid_u_loss, type="l", col="blue", lwd=2)
lines(dfu100.mask$valid_u_loss,lwd=2, lty="dashed", col="blue")

###

dftd50 = read.csv("../output_new/hybrid_both_deeper_tied_c50/results.txt")
dftd100 = read.csv("../output_new/hybrid_both_deeper_tied_c100/results.txt")
dftd200 = read.csv("../output_new/hybrid_both_deeper_tied_c200/results.txt")

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


# ------

# train loss
plot(dftd50$train_u_loss,type="l",lwd=2)
lines(dfud50$train_u_loss,col="red",lwd=2)
# valid loss
plot(dftd50$valid_u_loss,type="l",lwd=2)
lines(dfud50$valid_u_loss,col="red",lwd=2)

# train loss
plot(dftd100$train_u_loss,type="l",lwd=2)
lines(dfud100$train_u_loss,col="red",lwd=2)
# valid loss
plot(dftd100$valid_u_loss,type="l",lwd=2)
lines(dfud100$valid_u_loss,col="red",lwd=2)

# train loss
plot(sqrt(dftd50$train_i_loss),type="l",lwd=2)
lines(sqrt(dfid50$train_i_loss),col="red",lwd=2)
# valid loss
plot(sqrt(dftd50$valid_i_loss),type="l",lwd=2)
lines(sqrt(dfid50$valid_i_loss),col="red",lwd=2)


# train loss
plot(dftd100$train_i_loss,type="l",lwd=2)
lines(dfid100$train_i_loss,col="red",lwd=2)
# valid loss
plot(dftd100$valid_i_loss,type="l",lwd=2)
lines(dfid100$valid_i_loss,col="red",lwd=2)


# -------

dftd50 = read.csv("../output_new/hybrid_both_deeper_tied_c50/results.txt")
dftd100 = read.csv("../output_new/hybrid_both_deeper_tied_c100/results.txt")
dftd200 = read.csv("../output_new/hybrid_both_deeper_tied_c200/results.txt")

# ---



dfi50.deep = read.csv("../output_new/hybrid_item_deeper_c50/results.txt")
dfi100.deep = read.csv("../output_new/hybrid_item_deeper_c100/results.txt")
dfi200.deep = read.csv("../output_new/hybrid_item_deeper_c200/results.txt")

plot(dfi100.deep$valid_i_loss,type="l",lwd=2, ylim=c(0,1))
lines(dfi200.deep$valid_i_loss,lwd=2,col="blue")