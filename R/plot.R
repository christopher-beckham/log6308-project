par(mfrow=c(1,1))

k = 50

dfd_both = read.csv(paste("../output_new/hybrid_both_deeper_tied_c",k,"/results.txt",sep=""))

dfd_i = read.csv(paste("../output_new/hybrid_item_deeper_c",k,"/results.txt",sep=""))
plot(dfd_i$train_i_loss,type="l", lwd=2, xlab="epoch", ylab="loss")
lines(dfd_i$valid_i_loss,lty="dotted", lwd=2)
lines(dfd_both$train_i_loss, col="red", lwd=2)
lines(dfd_both$valid_i_loss, col="red", lty="dotted", lwd=2)
legend("topright", 
       legend=c("50-50(tr)", "50-50(vd)", "50-50(ws)(tr)", "50-50(ws)(vd)"),
       col=c("black", "black", "red", "red"),
       lty=c("solid", "dotted", "solid", "dotted"),
       lwd=c(2,2,2,2))

dfd_u = read.csv(paste("../output_new/hybrid_user_deeper_c",k,"/results.txt",sep=""))
plot(dfd_u$train_u_loss,type="l", lwd=2, xlab="epoch", ylab="loss")
lines(dfd_u$valid_u_loss,lty="dotted", lwd=2)
lines(dfd_both$train_u_loss, col="red", lwd=2)
lines(dfd_both$valid_u_loss, col="red", lty="dotted", lwd=2)

#dfd_both = read.csv("../output_new/hybrid_both_deeper_tied_c50/results.txt")
# item
#plot(dfd_both$train_i_loss,type="l", lwd=2, xlab="epoch", ylab="loss")
#lines(dfd_both$valid_i_loss,lty="dotted", lwd=2)

# ----------

df50 = read.csv("../output_new/hybrid_both_deeper_tied_c50/results.txt",header=T)
df100 = read.csv("../output_new/hybrid_both_deeper_tied_c100/results.txt",header=T)
df200 = read.csv("../output_new/hybrid_both_deeper_tied_c200/results.txt",header=T)
df300 = read.csv("../output_new/hybrid_both_deeper_tied_c300/results.txt",header=T)

plot(df50$valid_loss,type="l", lwd=2)
lines(df100$valid_loss,type="l",col="red", lwd=2)
lines(df200$valid_loss,type="l",col="green", lwd=2)
lines(df300$valid_loss,type="l",col="purple", lwd=2)
