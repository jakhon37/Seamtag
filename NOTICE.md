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

### OpenCV (BSD-3-Clause)

`src/main/resize.ts` and `src/main/prelabel.ts`: the bilinear and nearest resizes pre-labelling gives a model's tiles and takes its results back with, following OpenCV's `modules/imgproc/src/resize.cpp` (<https://github.com/opencv/opencv>) to the bit, as Seamline's own pre-labelling resizes them with OpenCV.

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

The README's screenshots show four photos from Wikimedia Commons. Each photographer dedicated theirs to the public domain under CC0 1.0, so no credit is required; this is where they come from:

- [A pedestrian crossing over Stadhouderskade, Amsterdam](https://commons.wikimedia.org/wiki/File:2023_Amsterdam_poto_-_Woman_are_walking_the_pedestrian_crossing_over_the_city_road_Stadhouderskade_in_early_Spring_sunlight;_free_download_photo_in_high_resolutions_by_Fons_Heijnbroek_,_street_photography_in_The_Netherlands,_CCO.jpg), by Fons Heijnsbroek
- [Erfurt Cafe Hilge 3](https://commons.wikimedia.org/wiki/File:Erfurt_Cafe_Hilge_3.jpg), by Ousw
- [Trent Buses (101 - YCH 890M) - 10 August 2025](https://commons.wikimedia.org/wiki/File:Trent_Buses_(101_-_YCH_890M)_-_10_August_2025.jpg), by Aethonatic
- [Market st. San Francisco.](https://commons.wikimedia.org/wiki/File:Market_st._San_Francisco._(48669724832).jpg), by Bernard Spragg. NZ

## AI models

The AI models aren't part of Seamtag. Each is downloaded when you ask for it, under its own licence, which AI → AI models shows: EfficientSAM and SAM2 are under the Apache License 2.0, YOLO-World under the GNU GPL v3.0.

## Installers and source

The installers carry this file, and beside it THIRD-PARTY-NOTICES.txt, with the licences of the libraries bundled into the app. The source code of each release is at <https://github.com/jakhon37/Seamtag>, under the release's tag.
