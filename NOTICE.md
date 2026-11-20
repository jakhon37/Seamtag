# Notices

## Seamtag

Copyright © 2026 Jakhongir Nodirov

Seamtag is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License only (GPL-3.0-only).

Seamtag is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Seamtag, in [LICENSE](LICENSE). If not, see <https://www.gnu.org/licenses/>.

## Where Seamtag began

Seamtag began in February 2026 as a fork of labelme (<https://github.com/wkentaro/labelme>), Copyright © Kentaro Wada and contributors, licensed under GPL-3.0-only. It was rewritten as an Electron app in 2026. The files below still contain code translated from labelme 7.9.1 into TypeScript, changed by Jakhongir Nodirov, 2026.

| File | Translated from labelme 7.9.1 |
|---|---|
| `src/shared/polygonize.ts` | `labelme/_automation/_geometry.py`: polygons, circles and rotated boxes from a mask |
| `src/main/detections.ts` | `labelme/_automation/_shape_builders.py`, `_suppression.py` and `_geometry.py`: shapes from detections, and the removal of repeats and of objects already drawn |
| `src/main/assist.ts` | `labelme/_automation/_ai_assist.py` and `_text_detection.py`: shapes proposed from clicks, a box or words |
| `src/shared/config.ts` | `labelme/_config/__init__.py`, `_migration.py` and `_shape_color.py`: checking, merging and migrating config files |
| `src/renderer/src/shapes.ts` | `labelme/_widgets/_canvas.py` and `labelme/_shape/_oriented_rectangle.py`: reshaping and turning a rotated box |
| `src/main/masks.ts` | `labelme/_shape/_merge.py`: merging masks (`mergeMasks`) |
| `src/renderer/src/Labeler.tsx` | `labelme/_widgets/_canvas.py`: selecting the next shape in an arrow's direction (`selectNeighbour`) |
| `src/shared/outline.ts` | `examples/instance_segmentation/labelme2coco.py`: circles and rectangles as polygons, for COCO and YOLO exports |

## Code from other projects

The files named below include code from other projects, under the licences that follow.

### LabelImg (MIT)

`src/shared/formats.ts`: reading and writing Pascal VOC, YOLO and CreateML files, ported from LabelImg 1.8.6 (<https://github.com/HumanSignal/labelImg>), `libs/pascal_voc_io.py`, `yolo_io.py`, `create_ml_io.py` and `labelFile.py`.

```text
Copyright (c) <2015-Present> Tzutalin

Copyright (C) 2013  MIT, Computer Science and Artificial Intelligence Laboratory. Bryan Russell, Antonio Torralba, William T. Freeman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### pycocotools (BSD-2-Clause)

`src/shared/rle.ts`: COCO's run-length encoded masks, ported from pycocotools 2.0.11's `common/maskApi.c` (<https://github.com/ppwwyyxx/cocoapi>).

```text
Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
```

### OpenAI CLIP (MIT)

`src/main/clip.ts`: the tokenizer for text prompts, translated from CLIP's `clip/simple_tokenizer.py` and `clip.tokenize` (<https://github.com/openai/CLIP>). `resources/clip/bpe_simple_vocab_16e6.txt.gz`: the first 48,895 lines of CLIP's vocabulary file, with this licence beside it in `resources/clip/LICENSE`.

```text
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### OpenCV (Apache-2.0 and BSD-3-Clause)

`src/main/resize.ts` and `src/main/prelabel.ts`: the bilinear and nearest resizes pre-labelling gives a model's tiles and takes its results back with, translated from OpenCV's `modules/imgproc/src/resize.cpp` (<https://github.com/opencv/opencv>) as it is in OpenCV 4.13.0 and 5.0.0, so a model Seamline exported proposes the same shapes in both apps. OpenCV is under the Apache License 2.0 since version 4.5.0; resize.cpp's older code keeps the BSD-3-Clause licence in its header. OpenCV's COPYRIGHT file, that header and OpenCV's LICENSE follow.

```text
Copyright (C) 2000-2022, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2023, OpenCV Foundation, all rights reserved.
Copyright (C) 2008-2016, Itseez Inc., all rights reserved.
Copyright (C) 2019-2023, Xperience AI, all rights reserved.
Copyright (C) 2019-2022, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
Copyright (C) 2022-2023, Southern University of Science And Technology, all rights reserved.
Copyright (C) 2023-2025, OpenCV AI, all rights reserved.

Third party copyrights are property of their respective owners.
```

```text
License Agreement
For Open Source Computer Vision Library

Copyright (C) 2000-2008, 2017, Intel Corporation, all rights reserved.
Copyright (C) 2009, Willow Garage Inc., all rights reserved.
Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistribution's of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistribution's in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * The name of the copyright holders may not be used to endorse or promote products
    derived from this software without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the Intel Corporation or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
```

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

### rtmlib and MMPose (Apache-2.0)

`src/main/rtmpose.ts`: finding a person's keypoints in a box with RTMPose, translated into TypeScript from rtmlib (<https://github.com/Tau-J/rtmlib>, at commit 5ea82df) and MMPose (<https://github.com/open-mmlab/mmpose>, at commit 759b39c). From rtmlib's `rtmlib/tools/pose_estimation/rtmpose.py` and `pre_processings.py`: the box's centre and scale, the warp of its part of the image to the model's input, and the keypoints read from the model's SimCC outputs and mapped back to the image. From MMPose's `mmpose/codecs/utils/post_processing.py` and `mmpose/models/pose_estimators/topdown.py`: each keypoint's score, the lower of its two maxima, and the same mapping back. Both are under the Apache License 2.0, whose text is given above under OpenCV, with these notices: MMPose's LICENSE, and the notice in the appendix of rtmlib's.

```text
Copyright 2018-2020 Open-MMLab. All rights reserved.
```

```text
Copyright 2018-2020 Open-MMLab.
```

### Pillow (MIT-CMU)

`src/main/resize.ts`: the bilinear resize of the images AI models are given, translated from Pillow's `src/libImaging/Resample.c` (<https://github.com/python-pillow/Pillow>).

```text
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh and contributors

Pillow is the friendly PIL fork. It is

    Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

Like PIL, Pillow is licensed under the open source MIT-CMU License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

### scikit-image (BSD-3-Clause)

`src/shared/polygonize.ts`: finding contours (`find_contours`) and simplifying them (`approximate_polygon`), translated from scikit-image (<https://github.com/scikit-image/scikit-image>), `skimage/measure/_find_contours_cy.pyx`, `_find_contours.py` and `_polygon.py`.

```text
Copyright: 2009-2022 the scikit-image team

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the University nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Photos in the README

The video at the top of the README and its screenshots show four photos from Wikimedia Commons. Each photographer dedicated theirs to the public domain under CC0 1.0, so no credit is required; this is where they come from:

- [A pedestrian crossing over Stadhouderskade, Amsterdam](https://commons.wikimedia.org/wiki/File:2023_Amsterdam_poto_-_Woman_are_walking_the_pedestrian_crossing_over_the_city_road_Stadhouderskade_in_early_Spring_sunlight;_free_download_photo_in_high_resolutions_by_Fons_Heijnbroek_,_street_photography_in_The_Netherlands,_CCO.jpg), by Fons Heijnsbroek
- [Erfurt Cafe Hilge 3](https://commons.wikimedia.org/wiki/File:Erfurt_Cafe_Hilge_3.jpg), by Ousw
- [Trent Buses (101 - YCH 890M) - 10 August 2025](https://commons.wikimedia.org/wiki/File:Trent_Buses_(101_-_YCH_890M)_-_10_August_2025.jpg), by Aethonatic
- [Market st. San Francisco.](https://commons.wikimedia.org/wiki/File:Market_st._San_Francisco._(48669724832).jpg), by Bernard Spragg. NZ

## AI models

The AI models aren't part of Seamtag. Each is downloaded when you ask for it, under its own licence, which AI → AI models shows: EfficientSAM and SAM2 are under the Apache License 2.0, YOLO-World under the GNU GPL v3.0.

RTMPose-m and RTMPose-l, which find people's keypoints, are OpenMMLab's, downloaded from <https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/> (or rtmlib's copy at <https://huggingface.co/Tau-J/RTMPose> when that doesn't answer) as zips, of which Seamtag keeps the ONNX file. Their weights carry no licence of their own: they're published with MMPose, under the Apache License 2.0 (<https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/LICENSE>). They were trained on a mix of public pose datasets (Body7), whose own terms vary.

## Installers and source

The installers carry this file, and beside it THIRD-PARTY-NOTICES.txt, with the licences of the libraries bundled into the app. The source code of each release is at <https://github.com/jakhon37/Seamtag>, under the release's tag.
