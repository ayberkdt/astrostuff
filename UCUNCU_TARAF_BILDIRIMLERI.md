# Üçüncü taraf bildirimleri

Bu depo üçüncü taraf kaynak kodu ya da ham verisi barındırmaz; üçüncü taraf veriden türetilmiş birkaç şekil içerir. Aşağıdaki bildirimler bazı fonksiyonların izlediği kaynak kodu ve şekillerin dayandığı veriler içindir.

## ERFA (Essential Routines for Fundamental Astronomy)

`kozmos.zaman.julyen`, `kozmos.temel.aci` ve `kozmos.koordinat.donusum` modüllerindeki fonksiyonlar, ERFA C kitaplığının kaynak dosyalarındaki algoritmaları ve katsayıları izleyerek Python'da yeniden yazılmıştır (`cal2jd.c`, `jd2cal.c`, `era00.c`, `gmst06.c`, `obl06.c`, `sepp.c`, `s2c.c`, `c2s.c`, `anp.c`, `anpm.c`, `icrs2g.c`, `hd2ae.c`, `ae2hd.c`, `erfam.h`). ERFA, IAU SOFA kitaplığından izinle türetilmiştir. Bu depo SOFA'nın kendisini değil, SOFA'dan türetilmiş bir kitaplığı kaynak almıştır.

ERFA kaynak dosyalarındaki telif ve lisans bildirimi:

```
Copyright (C) 2013-2023, NumFOCUS Foundation.
All rights reserved.

This library is derived, with permission, from the International
Astronomical Union's "Standards of Fundamental Astronomy" library,
available from http://www.iausofa.org.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1 Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

2 Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in
  the documentation and/or other materials provided with the
  distribution.

3 Neither the name of the Standards Of Fundamental Astronomy Board,
  the International Astronomical Union nor the names of its
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

Bu bildirimin burada bulunması bir hukuki değerlendirme değildir; ERFA'dan türetilen bölümlerin lisans yükümlülükleri depo sahibi tarafından ayrıca değerlendirilmelidir.

## Veri

Analizlerin kullandığı veri setleri depoya konmaz; çalışma anında kaynağından indirilir. Pantheon+SH0ES veri sürümünün deposunda lisans dosyası yoktur; veri Scolnic ve ark. (2022), Brout ve ark. (2022) ve Riess ve ark. (2022) makalelerine atıfla kullanılır.

## Şekillerde kullanılan tarama verileri

`belgeler/` klasöründeki bazı şekiller üçüncü taraf gözlem verisinden türetilmiştir. Depodaki hak saklama bu şekilleri kapsamaz; her biri aşağıdaki koşullara tabidir. Koşullar 25 Kasım 2026'da ilgili sayfalardan okunmuştur.

| Şekil | Veri | Koşul |
|---|---|---|
| `optik_kizilotesi.png` (sol panel) | Digitized Sky Survey 2, kırmızı bant | CC BY-NC-SA 4.0; kamu malı değildir |
| `optik_kizilotesi.png` (orta panel), `alev_bulutsusu_jhk.png`, `gercek_goruntu_dogrulamasi.png` | 2MASS görüntüleri ve Nokta Kaynak Kataloğu | Görüntü galerisindeki görüntüler kamu malıdır; teşekkür metni istenir |
| `optik_kizilotesi.png` (sağ panel) | WISE 12 µm | Teşekkür metni istenir; lisans ifadesi okunan sayfada yoktur |
| `hr_diyagrami.png` | Gaia DR3 | CC BY-NC 3.0 IGO |
| `gokyuzu_haritasi.png` | Parlak Yıldız Kataloğu, 5. baskı (HEASARC `bsc5p`) | Aşağıdaki nota bakınız |
| `hubble_diyagrami.png` | Pantheon+SH0ES veri sürümü | Okunmadı |

### Digitized Sky Survey

MAST'ın veri kullanım sayfası (`archive.stsci.edu/publishing/data-use`) DSS verisinin kamu malı olmadığını, bilimsel, eğitsel ve kâr amacı gütmeyen amaçlarla (web siteleri dahil) serbestçe kullanılabildiğini ve Creative Commons Atıf-GayriTicari-AynıLisanslaPaylaş 4.0 (CC BY-NC-SA 4.0) lisansına bağlı olduğunu bildirir. Bu yüzden `optik_kizilotesi.png` şekli de CC BY-NC-SA 4.0 ile paylaşılır ve ticari amaçla kullanılamaz.

Hak sahibi, kullanılan plakaya göre değişir: 1993 öncesi güney taramalarında UK SERC/PPARC (bugün STFC), 1993 sonrası güney taramalarında Anglo-Australian Observatory Board, POSS-II kuzey taramalarında California Institute of Technology, kalan malzemede Association of Universities for Research in Astronomy, Inc. Şekildeki alanın (δ = −1,9°) hangi plakadan geldiği SkyView başlığından okunmamıştır; hak sahibi bu dördünden biridir.

### 2MASS

IRSA'nın 2MASS sık sorulan sorular sayfası, görüntü galerilerindeki görüntülerin kamu malı olduğunu bildirir ve şu teşekkür metinlerini ister:

> This publication makes use of data products from the Two Micron All Sky Survey, which is a joint project of the University of Massachusetts and the Infrared Processing and Analysis Center/California Institute of Technology, funded by the National Aeronautics and Space Administration and the National Science Foundation.

> Atlas Image mosaic obtained as part of the Two Micron All Sky Survey (2MASS), a joint project of the University of Massachusetts and the Infrared Processing and Analysis Center/California Institute of Technology, funded by the National Aeronautics and Space Administration and the National Science Foundation.

Kamu malı ifadesi sayfada görüntü galerileri için geçer. Bu depodaki kesitler galeriden değil, SkyView'ın Atlas görüntülerinden yeniden örneklediği FITS dosyalarından üretilmiştir; ifadenin bu kesitleri de kapsadığı ayrıca doğrulanmamıştır.

### WISE

WISE Tüm Gökyüzü Sürümü açıklayıcı ekinin giriş bölümü şu teşekkür metnini ister:

> This publication makes use of data products from the Wide-field Infrared Survey Explorer, which is a joint project of the University of California, Los Angeles, and the Jet Propulsion Laboratory/California Institute of Technology, funded by the National Aeronautics and Space Administration.

### IRSA

Gaia ve 2MASS katalog sorguları IRSA üzerinden yapılmıştır. IRSA'nın "Acknowledging IRSA" sayfası şu metni ister:

> This research has made use of the NASA/IPAC Infrared Science Archive, which is funded by the National Aeronautics and Space Administration and operated by the California Institute of Technology.

### Gaia

ESA'nın Gaia lisans sayfası (`cosmos.esa.int/web/gaia-users/license`) Gaia verisinin CC BY-NC 3.0 IGO lisansıyla dağıtıldığını bildirir; ticari kullanım için ESA'nın arşiv koşullarına yönlendirir. Katalog verisi bu depoda barındırılmaz. Gaia'nın künye sayfası her veri sürümünün kendi teşekkür metnini verdiğini söyler; DR3'ün teşekkür metni okunmamıştır ve buraya eklenmemiştir.

### Parlak Yıldız Kataloğu

HEASARC'ın `bsc5p` sayfasındaki lisans notu, kataloğun yalnızca bilimsel araştırma amacıyla sunulduğunu, telif hakkının HEASARC'ta olmadığını ve ticari kullanım koşullarını açıklayamayacağını bildirir; basılı son sürümün (4. baskı) telif hakkı Yale University Observatory'dedir. Katalog bu depoda barındırılmaz; `gokyuzu_haritasi.png` katalogdaki konum ve kadirlerden çizilmiştir.

### SkyView ve Pantheon+

Görüntü kesitleri NASA SkyView servisinden indirilmiştir; SkyView'ın kendi teşekkür sayfasına ulaşılamamıştır (istek "File Not Found" döndürmüştür). Pantheon+SH0ES veri sürümünün kullanım koşulları okunmamıştır; veri bu depoda barındırılmaz, `hubble_diyagrami.png` ondan türetilmiştir.

## Yayınlar

Fonksiyonların dayandığı yayınlar `kaynakca/kaynaklar.bib` dosyasında listelenmiştir; denklemler bu yayınlardan alınmış, metinleri kopyalanmamıştır.
