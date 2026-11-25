# Üçüncü taraf bildirimleri

Bu depo üçüncü taraf kaynak kodu ya da verisi barındırmaz. Aşağıdaki bildirim, bazı fonksiyonların algoritma ve sayısal katsayılarının okunduğu kaynak için konmuştur.

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

## Yayınlar

Fonksiyonların dayandığı yayınlar `kaynakca/kaynaklar.bib` dosyasında listelenmiştir; denklemler bu yayınlardan alınmış, metinleri kopyalanmamıştır.
