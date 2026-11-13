# Licence texts node_modules lacks

`scripts/notices.mjs` writes these into THIRD-PARTY-NOTICES.txt, beside the texts it takes from node_modules.

- `onnxruntime-1.23.2/LICENSE` and `onnxruntime-1.23.2/ThirdPartyNotices.txt`: ONNX Runtime's licence, and its notices for the libraries built into it, from https://github.com/microsoft/onnxruntime at tag v1.23.2 (commit a83fc4d58cb48eb68890dd689f94f28288cf2278), fetched on 13 November 2026. The onnxruntime-node and onnxruntime-common packages carry neither. When onnxruntime-node moves to another version, fetch both files from that version's tag into a folder named after it; the script stops until they're there.
- `LGPL-3.0.txt`: the GNU Lesser General Public License version 3, as the FSF publishes it (https://www.gnu.org/licenses/lgpl-3.0.txt), copied from Ubuntu's base-files (`/usr/share/common-licenses/LGPL-3`). sharp's libvips packages name the licence but don't carry it.
- `clipper-lib/BSL-1.0.txt`: the Boost Software License 1.0 (https://www.boost.org/LICENSE_1_0.txt), which clipper.js names, as zlib carries it in `contrib/dotzlib/LICENSE_1_0.txt`.
- `clipper-lib/jsbn-LICENSE`: the licence of Tom Wu's JSBN, part of which is in clipper.js. clipper.js points to http://www-cs-students.stanford.edu/~tjw/jsbn/LICENSE; this is that text as the jsbn 1.1.0 package (https://github.com/andyperlitch/jsbn) carries it.
