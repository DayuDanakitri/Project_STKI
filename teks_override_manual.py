"""
TEKS LENGKAP PASAL 45, 45A, 45B -- OVERRIDE MANUAL
=====================================================
Pasal-pasal ini di PDF aslinya terpotong oleh pergantian halaman (pola
"(2) Perbuatan . . ." menandakan teks bersambung ke halaman berikutnya),
sehingga ekstraksi otomatis berbasis jangkar regex di fase1_indexing.py
hanya menangkap ayat (1) saja, bukan keseluruhan pasal sampai ayat (11).

Teks di bawah ini disusun manual dari pembacaan langsung dokumen PDF
(halaman demi halaman) untuk memastikan seluruh ayat pidana tertangkap
utuh -- ini PENTING karena Pasal 45 adalah pasal sanksi utama yang
dirujuk ground truth evaluasi (Pasal 27 -> sanksi di Pasal 45, dst).

Dipakai oleh fase1_indexing.py sebagai override setelah ekstraksi
otomatis, BUKAN pengganti seluruh proses ekstraksi.
"""

TEKS_OVERRIDE_MANUAL = {
    "Pasal 45": """Pasal 45
(1) Setiap Orang yang dengan sengaja dan tanpa hak menyiarkan, mempertunjukkan, mendistribusikan, mentransmisikan, dan/atau membuat dapat diaksesnya Informasi Elektronik dan/atau Dokumen Elektronik yang memiliki muatan yang melanggar kesusilaan untuk diketahui umum sebagaimana dimaksud dalam Pasal 27 ayat (1) dipidana dengan pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah).
(2) Perbuatan sebagaimana dimaksud pada ayat (1) tidak dipidana dalam hal:
    a. dilakukan demi kepentingan umum;
    b. dilakukan untuk pembelaan atas dirinya sendiri; atau
    c. Informasi Elektronik dan/atau Dokumen Elektronik tersebut merupakan karya seni, budaya, olahraga, kesehatan, dan/atau ilmu pengetahuan.
(3) Setiap Orang yang dengan sengaja dan tanpa hak mendistribusikan, mentransmisikan, dan/atau membuat dapat diaksesnya Informasi Elektronik dan/atau Dokumen Elektronik yang memiliki muatan perjudian sebagaimana dimaksud dalam Pasal 27 ayat (2) dipidana dengan pidana penjara paling lama 10 (sepuluh) tahun dan/atau denda paling banyak Rp10.000.000.000,00 (sepuluh miliar rupiah).
(4) Setiap Orang yang dengan sengaja menyerang kehormatan atau nama baik orang lain dengan cara menuduhkan suatu hal, dengan maksud supaya hal tersebut diketahui umum dalam bentuk Informasi Elektronik dan/atau Dokumen Elektronik yang dilakukan melalui Sistem Elektronik sebagaimana dimaksud dalam Pasal 27A dipidana dengan pidana penjara paling lama 2 (dua) tahun dan/atau denda paling banyak Rp400.000.000,00 (empat ratus juta rupiah).
(5) Ketentuan sebagaimana dimaksud pada ayat (4) merupakan tindak pidana aduan yang hanya dapat dituntut atas pengaduan korban atau orang yang terkena tindak pidana dan bukan oleh badan hukum.
(6) Dalam hal perbuatan sebagaimana dimaksud pada ayat (4) tidak dapat dibuktikan kebenarannya dan bertentangan dengan apa yang diketahui padahal telah diberi kesempatan untuk membuktikannya, dipidana karena fitnah dengan pidana penjara paling lama 4 (empat) tahun dan/atau denda paling banyak Rp750.000.000,00 (tujuh ratus lima puluh juta rupiah).
(7) Perbuatan sebagaimana dimaksud pada ayat (4) tidak dipidana dalam hal:
    a. dilakukan untuk kepentingan umum; atau
    b. dilakukan karena terpaksa membela diri.
(8) Setiap Orang yang dengan sengaja dan tanpa hak mendistribusikan dan/atau mentransmisikan Informasi Elektronik dan/atau Dokumen Elektronik, dengan maksud untuk menguntungkan diri sendiri atau orang lain secara melawan hukum, memaksa orang dengan ancaman kekerasan untuk:
    a. memberikan suatu barang, yang sebagian atau seluruhnya milik orang tersebut atau milik orang lain; atau
    b. memberi utang, membuat pengakuan utang, atau menghapuskan piutang,
    sebagaimana dimaksud dalam Pasal 27B ayat (1) dipidana dengan pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah).
(9) Dalam hal perbuatan sebagaimana dimaksud pada ayat (8) dilakukan dalam lingkungan keluarga, penuntutan pidana hanya dapat dilakukan atas aduan.
(10) Setiap Orang yang dengan sengaja dan tanpa hak mendistribusikan dan/atau mentransmisikan Informasi Elektronik dan/atau Dokumen Elektronik, dengan maksud untuk menguntungkan diri sendiri atau orang lain secara melawan hukum, dengan ancaman pencemaran atau dengan ancaman akan membuka rahasia, memaksa orang supaya:
    a. memberikan suatu barang yang sebagian atau seluruhnya milik orang tersebut atau milik orang lain; atau
    b. memberi utang, membuat pengakuan utang, atau menghapuskan piutang,
    sebagaimana dimaksud dalam Pasal 27B ayat (2) dipidana dengan pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah).
(11) Tindak pidana sebagaimana dimaksud pada ayat (10) hanya dapat dituntut atas pengaduan korban tindak pidana.""",

    "Pasal 45A": """Pasal 45A
(1) Setiap Orang yang dengan sengaja mendistribusikan dan/atau mentransmisikan Informasi Elektronik dan/atau Dokumen Elektronik yang berisi pemberitahuan bohong atau informasi menyesatkan yang mengakibatkan kerugian materiel bagi konsumen dalam Transaksi Elektronik sebagaimana dimaksud dalam Pasal 28 ayat (1) dipidana dengan pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah).
(2) Setiap Orang yang dengan sengaja dan tanpa hak mendistribusikan dan/atau mentransmisikan Informasi Elektronik dan/atau Dokumen Elektronik yang sifatnya menghasut, mengajak, atau memengaruhi orang lain sehingga menimbulkan rasa kebencian atau permusuhan terhadap individu dan/atau kelompok masyarakat tertentu berdasarkan ras, kebangsaan, etnis, warna kulit, agama, kepercayaan, jenis kelamin, disabilitas mental, atau disabilitas fisik sebagaimana dimaksud dalam Pasal 28 ayat (2) dipidana dengan pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah).
(3) Setiap Orang yang dengan sengaja menyebarkan Informasi Elektronik dan/atau Dokumen Elektronik yang diketahuinya memuat pemberitahuan bohong yang menimbulkan kerusuhan di masyarakat sebagaimana dimaksud dalam Pasal 28 ayat (3) dipidana dengan pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah).""",

    "Pasal 45B": """Pasal 45B
Setiap Orang yang dengan sengaja dan tanpa hak mengirimkan Informasi Elektronik dan/atau Dokumen Elektronik secara langsung kepada korban yang berisi ancaman kekerasan dan/atau menakut-nakuti sebagaimana dimaksud dalam Pasal 29 dipidana dengan pidana penjara paling lama 4 (empat) tahun dan/atau denda paling banyak Rp750.000.000,00 (tujuh ratus lima puluh juta rupiah).""",
}
