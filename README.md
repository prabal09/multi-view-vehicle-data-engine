Built an open-vocabulary perception pipeline using Grounding DINO and SAM2 to extract ~2000 unique vehicle tracks from 8 hours of video across 5 calibrated viewpoints.
Localized each vehicle into a shared bird's-eye-view frame using homography based purely geometric cross-view association. Associated detections across views by fusing BEV position with appearance to produce per-vehicle multi-view frame stacks.



    Vehicle Type	Vehicle Model	     Track	frame-start	frame-end	time-start	time-end            
      SUV	         Cadillac Escalade
c0	                               	c0-5	   590	          595       0.19	    0.20
c1			                            c1-1	   7640	          7646	    4.13	    4.15
c2			                            c2-0	   5495	          5510	    3.03	    3.05
c3				                          c3-3     7486	          7495	    4.09	    4.11
c4			                            c4-2	   8902	          8950	    4.57	    4.59
