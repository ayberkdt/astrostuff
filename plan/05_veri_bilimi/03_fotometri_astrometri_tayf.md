# Fotometri, astrometri ve tayf ölçümü

Hedef modüller: `fotometri/`, `astrometri/`, `tayf/`.

Girdi indirgenmiş görüntüdür ([02_goruntu_indirgeme_hatti.md](02_goruntu_indirgeme_hatti.md)); çıktı kalibre edilmiş katalog ya da tayftır.

## 1. Kaynak saptama

- Eşleşmiş süzgeç: görüntünün PSF ile evrişimi nokta kaynak için en yüksek S/N'yi verir; eşik arka plan RMS'inin katıdır.
- Yıldız bulucu: evrişimli görüntüde yerel en büyükler, keskinlik ve yuvarlaklık ölçütleriyle kozmik ışın ve yaygın kaynak ayıklama [Stetson1987].
- Bölütleme: eşik üstü bağlı piksel grupları, en küçük alan koşulu; çok eşikli ayrıştırma (deblending) [BertinArnouts1996].
- Tamlık ve saflık: yapay yıldız ekleme deneyiyle kadire karşı ölçülür.

Kâhinler: `photutils` (DAOStarFinder, bölütleme), `sep` [Barbary2016].

## 2. Merkez ve biçim ölçümü

- Merkez: ağırlıklı ilk moment, kenar toplamlarına Gauss uydurma, 2B PSF uydurma. Astrometrik duyarlık yaklaşık `FWHM / (S/N)` mertebesindedir; katsayı ve türetim kaynağıyla yazılır [King1971].
- İkinci momentler: FWHM, eliptiklik, konum açısı; yıldız-gökada ayrımı.

## 3. Açıklık fotometrisi

- Dairesel ve eliptik açıklık; piksel-daire kesişiminin tam alan hesabı (alt piksel örnekleme değil, geometrik çözüm).
- Yerel gökyüzü: halka içinde sigma kırpmalı kestirim.
- Belirsizlik: CCD denklemi [MerlineHowell1995].
- Açıklık düzeltmesi: küçük açıklık akısının büyüme eğrisiyle toplam akıya taşınması.
- Kron açıklığı: ilk moment yarıçapına bağlı uyarlamalı eliptik açıklık; gökadalar için [Kron1980].
- En iyi (ağırlıklı) fotometri: PSF ağırlıklı toplam; soluk kaynaklarda açıklık fotometrisinden yüksek S/N [Naylor1998].

## 4. PSF fotometrisi

- PSF modeli: analitik (Gauss, Moffat) + deneysel artık tablosu; alan boyunca değişim [Stetson1987].
- PSF yıldızlarının seçimi: parlak, yalıtılmış, doymamış.
- Uydurma: konum ve akı için doğrusal olmayan en küçük kareler; kalabalık alanda komşu grupların eşzamanlı uydurulması.
- Yineleme: uydur, çıkar, artıkta yeni kaynak ara.
- Kalite ölçütleri: `χ`, keskinlik; artık görüntü denetimi.

## 5. Fotometrik kalibrasyon

- Aygıt kadiri: `m_aygıt = −2,5 log10(akı / t)`.
- **Mutlak (tüm gökyüzü) fotometri:** `m_std = m_aygıt + ZP − k X + c (renk)`; standart yıldızlarla sıfır noktası, sönümleme katsayısı ve renk terimi birlikte çözülür [Landolt1992; Bessell2005].
- **Diferansiyel fotometri:** aynı karedeki karşılaştırma yıldızlarına göre; atmosferin birinci mertebe etkisi düşer. Topluluk fotometrisi: çok sayıda karşılaştırma yıldızının ağırlıklı birleşimi.
- **Katalog kalibrasyonu:** alan yıldızlarının Gaia, Pan-STARRS ya da SDSS kadirleriyle eşlenmesi; sistem dönüşümü ve renk terimi sağlam doğrusal uyumla [Vallenari2023; Fukugita1996].
- Sistematikler: düz alan hatası, ikinci mertebe sönümleme, perde süresi, doğrusallık; her biri için tanı grafiği.

## 6. Astrometrik çözüm

- **Kör çözüm:** dörtlü yıldız asterizmlerinin geometrik karma kodu ve dizin araması [Lang2010]. Bu algoritma yeniden yazılabilir ama dizin dosyaları büyüktür; ilk aşamada `astrometry.net` ya da `twirl` kullanılır, yaklaşık konum bilinen durum için kendi çözücümüz yazılır.
- **Yaklaşık konum bilinen durum:** saptanan kaynaklar ile Gaia kataloğunun TAN izdüşümünde üçgen eşlemesi; ardından doğrusal (6 parametre) ve polinom bozulma uyumu.
- Referans katalog: Gaia DR3; yıldızlar gözlem dönemine öz hareketle taşınır [Vallenari2023].
- Çözüm kalitesi: artıkların RMS'i, alan boyunca desen; diferansiyel renk kırılması.
- Küçük cisim astrometrisi: hareketli kaynakta poz ortası zamanı ve iz merkezi.

## 7. Tayf indirgeme

Uzun yarık ve echelle için ortak sıra:

1. Görüntü indirgeme (bias, karanlık, düz alan; tayfsal düz alanda lambanın sürekliliği normalleştirilir).
2. **İz bulma:** dağılma ekseni boyunca tayfın uzamsal merkezinin polinomla izlenmesi.
3. **Arka plan:** yarık boyunca gökyüzü bölgelerinden dalga boyu sütunu başına model.
4. **Çıkarım:** kutu toplamı ve en iyi çıkarım. En iyi çıkarım uzamsal profili ağırlık olarak kullanır, S/N'yi artırır ve kozmik ışınları profil uyumsuzluğundan ayıklar [Horne1986].
5. **Dalga boyu kalibrasyonu:** ark lambası çizgilerinin saptanması, çizgi listesiyle eşleme, piksel → dalga boyu polinomu; artık RMS'i kalite ölçütüdür. Gökyüzü çizgileriyle sıfır noktası düzeltmesi. Hava ve boşluk dalga boyu dönüşümü havanın kırılma indisi formülüne dayanır [Ciddor1996].
6. **Akı kalibrasyonu:** tayfsal standart yıldızla duyarlılık fonksiyonu; atmosferik sönümleme düzeltmesi [OkeGunn1983].
7. **Tellürik düzeltme** ve barisentrik düzeltme [WrightEastman2014].

Kâhinler: `specreduce`, `specutils`, PypeIt.

## 8. Tayf çözümlemesi

- Süreklilik normalleştirme: yinelemeli polinom ya da spline, çizgi maskeleme.
- Çizgi ölçümü: Gauss, Lorentz, Voigt uydurma; eşdeğer genişlik, akı, merkez, FWHM ve belirsizlikleri.
- **Kırmızıya kayma:** salma çizgisi eşleme; şablonla çapraz ilinti ve tepe uydurma, `r` istatistiğiyle güven ölçüsü [TonryDavis1979]; şablon `χ²` taraması.
- Dikine hız: log-dalga boyu ızgarasında çapraz ilinti.
- Tayf sınıflaması: çizgi oranları ve şablon eşleme.
- Tayfların birleştirilmesi ve yeniden örneklenmesi: akı ve varyans korunumu.

Kırmızıya kayma ölçümü Hubble diyagramının yatay eksenini verir; SDSS tayfları üzerinde yayımlanmış değerlerin yeniden elde edilmesi bu modülün uçtan uca testidir [York2000].

## 9. Doğrulama

- Sentetik yıldız alanında açıklık ve PSF fotometrisi girilen akıyı yansız geri verir; artıklar belirsizlik kestirimiyle tutarlıdır.
- Aynı görüntüde `photutils` ve `sep` ile kaynak listesi ve akı karşılaştırması.
- Astrometrik çözüm `astrometry.net` çözümüyle karşılaştırılır; fark Gaia artıklarının altındadır.
- Bir açık kümenin renk-kadir diyagramı yayımlanmış fotometriyle karşılaştırılır.
