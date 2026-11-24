- TENO's shared-table setup packs each stencil shape's record on the device and copies only those to the host,
  not every slot of the batch's arrays: less host transfer at setup, and compute-sanitizer's initcheck no longer
  reports the copy of entries the setup never wrote (#1077).
