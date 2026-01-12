# morphint
[morphint](images/Morphint.jpg)
### Purpose
Interpolation between 2D brain sections within a brain volume.

### About

Morphint uses ANTs to calculate nonlinear diffeomorphisms between coronal sections in a user provided 3D volume. To interpolate missing sections between two acquired sections (posterior and anterior), 
the diffeomorphism is scaled in the forward and inverse direction and applied to the posterior and anterior sections, respectively. The warped sections are averaged using distance-weighted linear interpolation.

### Useage
```
from morphint.morphtint import morphint

target_resolution=0.5 # 0.5mm isottropic output resolution
resolution_list = [2,1,0.5] # List of resolutions to use for ANTs alignment between sections

morphint("/path/to/brain/volume.nii.gz"
         "/path/to/outputs/",
        target_resolution,
        resolution_list
      )
        

