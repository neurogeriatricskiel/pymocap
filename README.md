# PyMoCap: Python for Motion Capture

In this repository a collection of scripts will be made publicly available for the analysis of motion capture data.  
At the Neurology department of the University Hospital Kiel (UKSH) we are working with both 
1. opto-electronic stereophotogrammetric systems, based on passive retro-reflective markers, hereafter referred to as optical motion capture (OMC) systems, and 
2. wearable inertial measurement unit (IMU) systems, usually containing a 3-axis accelerometer and 3-axis gyroscope.  

In-lab recordings are used to validate IMU-based algorithms where reference values are obtained from the OMC systems. In the long run, the IMU-based algorithms are to be used in the home environment to gain insight in real-world gait.

## Processing of Optical Motion Capture Data

OMC data generally suffer from gaps in the marker trajectories due to marker occlusion or markers falling off. A first step in processing of marker data is therefore to fill any gaps in the trajectories (Federolf, 2013; Gløersen and Federolf, 2016). Next, marker data are low-pass filtered (in a forward and backward pass to accout for any delay due to filtering) with a 4th order Butterworth filter at a cut-off frequency, $f_{\mathsf{cut}}$, of 5 Hz (Godfrey *et al*., 2008).  
Marker data are then aligned with the main direction of walking, and data are passed through the methods to detecting ICs (O'Connor *et al*., 2007) and FCs (Zeni Jr *et al*., 2008).

## References
- Federolf P. A. (2013). A novel approach to solve the "missing marker problem" in marker-based motion analysis that exploits the segment coordination patterns in multi-limb motion data. PloS one, 8(10), e78689. <a href="https://doi.org/10.1371/journal.pone.0078689">https://doi.org/10.1371/journal.pone.0078689</a>
- Gløersen, Ø., & Federolf, P. (2016). Predicting Missing Marker Trajectories in Human Motion Data Using Marker Intercorrelations. PloS one, 11(3), e0152616. <a href="https://doi.org/10.1371/journal.pone.0152616">https://doi.org/10.1371/journal.pone.0152616</a>
- Godfrey, A., Conway, R., Meagher, D., & OLaighin, G. (2008). Direct measurement of human movement by accelerometry. Medical engineering & physics, 30(10), 1364–1386. <a href="https://doi.org/10.1016/j.medengphy.2008.09.005">https://doi.org/10.1016/j.medengphy.2008.09.005</a>
- O'Connor, C. M., Thorpe, S. K., O'Malley, M. J., & Vaughan, C. L. (2007). Automatic detection of gait events using kinematic data. Gait & posture, 25(3), 469–474. <a href="https://doi.org/10.1016/j.gaitpost.2006.05.016">https://doi.org/10.1016/j.gaitpost.2006.05.016</a>
- Zeni, J. A., Jr, Richards, J. G., & Higginson, J. S. (2008). Two simple methods for determining gait events during treadmill and overground walking using kinematic data. Gait & posture, 27(4), 710–714. <a href="https://doi.org/10.1016/j.gaitpost.2007.07.007">https://doi.org/10.1016/j.gaitpost.2007.07.007</a>