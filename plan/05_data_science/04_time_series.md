# Zaman serisi çözümlemesi

Hedef modül: `zamanserisi/`.

Astronomik zaman serileri eşit aralıklı değildir (gündüz, hava, mevsim boşlukları), gürültüleri nokta nokta değişir ve çoğu kez ilintilidir. Yöntemler bu üç özelliğe göre seçilir.

## 1. Veri modeli ve ön işleme

- Işık eğrisi kabı: zaman (ölçeği ve referansı belirtilmiş, tercihen `BJD_TDB`), akı ya da kadir, belirsizlik, kalite bayrağı, bant [Eastman2010].
- Aykırı değer ayıklama: hareketli medyana göre sigma kırpma; geçiş gibi gerçek sinyali kesmemek için tek yönlü kırpma.
- **Eğilim giderme:** hareketli medyan, çift ağırlıklı süzgeç, Savitzky-Golay, spline, Gauss süreci. Pencere genişliği aranan sinyalin süresinden uzun seçilir; aksi halde sinyal bastırılır. Kâhin: `wotan`.
- Sistematik giderme: ortak eğilimlerin temel bileşenlerle modellenmesi (uzay teleskobu verisinde); piksel düzeyinde bağlanım.
- Kutulama ve faz katlama.

## 2. Dönem arama

**Lomb-Scargle periyodogramı.** Eşit aralıklı olmayan veride sinüs uydurmanın `χ²` iyileşmesi olarak yorumlanır [Lomb1976; Scargle1982]. Uygulama ayrıntıları VanderPlas'ın derlemesini izler [VanderPlas2018]:

- Yüzen ortalama (genelleştirilmiş) biçim ve ölçüm hatası ağırlıkları.
- Frekans ızgarası: çözünürlük `1/T`'nin kesri (aşırı örnekleme katsayısı 5 ve üstü); üst sınır seçimi. Eşit aralıklı olmayan veride Nyquist frekansı iyi tanımlı değildir.
- Pencere fonksiyonu ve takma adlar: günlük (`±1 gün⁻¹`), yıllık ve örnekleme kaynaklı tepeler. Pencere periyodogramı her analizde çizilir.
- Yanlış alarm olasılığı: analitik yaklaşımlar ve bootstrap.
- Çok terimli (Fourier serisi) uzantı: sinüs dışı biçimler için.
- Hızlı algoritma: `O(N log N)` yaklaşımı; önce doğrudan `O(N·N_f)` biçim yazılır.

**Faz saçılımı yöntemleri.** Katlanmış ışık eğrisinin saçılımını en küçükleyen dönem; sinüs dışı biçimlerde (örten çift) güçlüdür [Stellingwerf1978].

**Dönemin inceltilmesi ve belirsizliği:** tepe çevresinde model uydurma; MCMC ile sonsal dağılım. Periyodogram tepesinin genişliği belirsizlik değildir.

Kâhin: `astropy.timeseries.LombScargle`.

## 3. Geçiş arama

- **Kutu en küçük kareler (BLS):** dönem, süre ve evre ızgarasında kutu biçimli çukur araması; sinyal artık oranı ve sinyal saptama verimi istatistikleri [Kovacs2002].
- **Geçiş en küçük kareler (TLS):** kutu yerine kenar kararmalı gerçekçi geçiş biçimi; küçük gezegenlerde daha duyarlıdır [HippkeHeller2019].
- Dönem ızgarası: frekansta düzgün değil, yıldız yoğunluğuna bağlı en iyi örnekleme.
- Doğrulama sınamaları: tek-çift geçiş derinliği farkı, ikincil tutulma, merkez kayması; örten çift yanlış pozitiflerinin ayıklanması.

Kâhinler: `astropy.timeseries.BoxLeastSquares`, `transitleastsquares`.

## 4. Model uydurma

- Geçiş modeli: yarıçap oranı, ölçeklenmiş yarı büyük eksen, eğim, kenar kararma katsayıları, geçiş ortası zamanı [MandelAgol2002; Winn2010]. Kâhin: `batman` [Kreidberg2015].
- Uzun pozlarda modelin poz süresi üzerinden integrali.
- Zonklayan değişenler: Fourier ayrışımı, genlik ve evre oranları; Sefeid ve RR Lyrae sınıflaması.
- Örten çiftler: en küçük zamanları ve O-C diyagramı; dönem değişimi.
- Süpernova ışık eğrileri: şablon uydurma `sncosmo` üzerinden [Guy2007].

## 5. İlintili gürültü

- Kırmızı gürültünün varlığında beyaz gürültü varsayımı belirsizlikleri küçük gösterir.
- Tanı: kutulanmış artıkların RMS'inin `1/√N` ile ölçeklenip ölçeklenmediği.
- Gauss süreci ile gürültünün ve sinyalin birlikte modellenmesi [RasmussenWilliams2006; ForemanMackey2017].
- Stokastik değişkenlik: yapı fonksiyonu, sönümlü rastgele yürüyüş (etkin gökada çekirdekleri) [Ivezic2020, Böl. 10].

## 6. Diğer yöntemler

- Dalgacık dönüşümü: zamanla değişen dönemler.
- Öz ilinti fonksiyonu: yıldız dönme dönemi.
- Bayesçi bloklar: patlama ve parlama saptama [Ivezic2020, Böl. 5].
- Geçici kaynak sınıflaması: öznitelik çıkarımı ve denetimli öğrenme (ZTF ve Rubin uyarı akışları) [Bellm2019].

## 7. Doğrulama

- Sentetik ışık eğrisi: bilinen dönem, genlik, gerçekçi örnekleme ve gürültü. Geri kazanım oranı S/N'ye ve gözlem sayısına karşı ölçülür.
- Enjeksiyon-geri kazanım: gerçek ışık eğrisine yapay geçiş eklenir; saptama verimi haritası çıkarılır.
- Yayımlanmış bir TESS gezegeninin dönemi, derinliği ve süresi yeniden elde edilir [Ricker2015].
- OGLE ya da Gaia Sefeidlerinin katalog dönemleri yeniden bulunur.
- Periyodogram değerleri `astropy` ile aynı ızgarada `1e-10` düzeyinde uyuşur.
