Multivariate classification algorithm comparison (applied to HEP)


The focus of this project is to study a *new physics* process. We briefly introduce the Standard
Model (SM), a theory that explains a great range of natural phenomena, but is not yet a complete 
description of nature. To explain these kind of out-of-SM phenomena we use extensions to the SM; in
the final state of the main process studied in this project there are particles that could be explained
by an extension to the SM called supersymmetry.

To study these phenomena we need a very specific experimental setup, comprehended by the Large Hadron
Collider and the Compact Muon Solenoid ; this setup and its subcomponents will be explained thoroughly, so
will be the event identification and reconstruction processes that provide us with the data used to
carry out our analysis.

After explaining the data collection we then will explain in detail the studied supersymmetric process, define
our signal and background and using Monte Carlo simulations we will establish our cinematic *regions of interest*.
The background in these regions is mostly made up from errors in the determination of the source of the final state 
particles. It is then deemed possible to train an algorithm that does a better classification of the final-state-particle's 
origin and minimizes the error in order to diminish the experimental background.

The obvious next step is to study a multivariate method that will help with this particle tagging. We will try to 
train a the algorithm so that we get a classification with a lower error rate than the original classification. Our 
choice of algorithm for this purpose is going to be a neural network.

To conclude this study we then will apply and compare the particle source predictions by the original classification and 
our trained algorithm in the regions of interest so to quantitatively evaluate the improvement in the predictions and the
measures of our process.