num_items = 65133
ks = c(5, 10, 50, 100, 200, 300)
ms = c(5, 50, 100, 500, 1000)
plot(ks, log(ks*num_items), type="l", ylim=c(10,20), xlab="k (bottleneck)", ylab="log(num params)"); points(ks, log(ks*num_items))
rainbows = rainbow(length(ms))
for(i in 1:length(ms)) {
  m = ms[i]
  np = (num_items*m)+(m*ks)
  lines(ks, log(np), col=rainbows[i]); points(ks, log(np), col=rainbows[i])
}
legend("bottomright", legend=paste("m =",ms), col=rainbows, lty=rep("solid",length(ms)),cex=0.7)
