# Optik ve kızılötesi gözlem: uygulanan hat ve sonraki adımlar

Modüller: `arsiv/` (tap, gaia, skyview, onbellek), `io/fits.py`, `gozlem/` (dedektor, planlama), `indirgeme/`, `fotometri/`, `gorsel/`, `isinim/kizilotesi.py`.

## Uygulanan hat

Arşivden görüntüye ve kataloğa giden zincir 25 Kasım 2026'da uçtan uca çalıştırılmıştır:

1. **Erişim.** TAP istemcisi ADQL sorgusunu çalıştırır ve sonucu CSV olarak önbelleğe yazar [TAP2019]; SkyView istemcisi verilen konumda görüntü kesiti indirir [SkyView2026]. Her indirme adresi, SHA-256 özeti ve zamanıyla kayda geçer.
2. **Okuma.** FITS okuyucu birincil görüntü birimini standarda göre çözer [FITS2018]; `astropy.io.fits` ile bit düzeyinde karşılaştırılmıştır.
3. **Arka plan.** Hücre ızgarasında kırpmalı kip kestirimi, ortanca süzgeci ve spline enterpolasyonu [SExtractorBelge2026].
4. **Saptama.** Arka plan üstündeki bağlı piksel grupları; isteğe bağlı Gauss yumuşatma.
5. **Ölçüm.** Dairesel açıklık fotometrisi, halkadan yerel gökyüzü.
6. **Görselleştirme.** asinh germe ve renk koruyan RGB bileşimi [Lupton2004].

Katalog tarafında Gaia DR3'ten 50 pc içindeki 31 298 yıldız indirilmiş ve HR diyagramı kurulmuştur [Babusiaux2018; Vallenari2023].

## Servislerin durumu

25 Kasım 2026'da denenmiş ve sonuçları gözlenmiştir:

- **IRSA TAP** (`irsa.ipac.caltech.edu/TAP/sync`): çalışıyor; `gaia_dr3_source` ve 2MASS `fp_psc` tabloları sorgulandı, CSV döndü.
- **NASA SkyView** (`skyview.gsfc.nasa.gov/current/cgi/runquery.pl`): çalışıyor; DSS2, 2MASS ve WISE kesitleri FITS olarak indirildi.
- **Gaia AIP aynası** (`gaia.aip.de/tap/sync`): çalışıyor; VOTable döndürüyor.
- **ESA Gaia arşivi** (`gea.esac.esa.int/tap-server/tap/sync`): sorguya açılış sayfası (HTML) döndürdü; uç nokta değişmiş olabilir.
- **CDS** (TAPVizieR, hips2fits): bağlantı kurulamadı ya da 404 döndü; nedeni incelenmedi.

Hipparcos gibi parlak yıldız katalogları VizieR'da durduğu için indirilememiştir. Gaia örneklemi çok parlak yıldızları içermez; adlandırılmış parlak yıldızlar kataloğu eksik kalmıştır.

## Optik ve kızılötesi karşılaştırması

Alev Bulutsusu (NGC 2024) çevresinde 0,3 derecelik alan üç bantta işlenmiştir (`ornekler/optik_kizilotesi.py`). Aynı saptama ayarlarıyla (4σ, en az 6 piksel) bulunan kaynak sayıları:

- DSS2 Red (0,64 µm): 17
- 2MASS K (2,16 µm): 132
- WISE 12 µm: 15

Sayılar aynı algoritmanın üç görüntüdeki çıktısıdır; tamlık ve yanlış saptama oranı bu alanda ölçülmemiştir. Parlak bulutsu yapısının bir kısmı da "kaynak" olarak sayılır; optik ve 12 µm görüntülerinde bu etki daha büyüktür. Yöntemin tamlığı ve konum doğruluğu sentetik yıldız alanlarında sınanmıştır (`tests/birim/test_fotometri.py`).

12 µm görüntüsünün ortasındaki koyu bölge görüntüdeki en düşük değerli piksellerdir; dedektör doyması olduğu sanılmaktadır, başlıktan ya da WISE belgelerinden doğrulanmamıştır.

## Kızılötesine özgü konular

Uygulananlar:

- 2MASS J, H, Ks ve WISE W1-W4 bantlarının Vega sıfır noktaları; kadir-akı dönüşümü ve AB farkı [Cohen2003; Wright2010].
- Wien yasasıyla bandın en duyarlı olduğu ısıl kaynak sıcaklığı [CODATA2022].

Yapılacaklar:

- **Gökyüzü çıkarma.** Yakın kızılötesinde gökyüzü parlak ve dakikalar içinde değişkendir. Kaydırmalı (dither) karelerden hareketli ortanca gökyüzü karesi kurulur ve her kareden çıkarılır.
- **Isıl arka plan.** 3 µm'nin ötesinde teleskop ve atmosferin kendi ışıması baskındır; kısa poz, hızlı okuma ve nodlama gerekir. Arka plan foton gürültüsü `gozlem.dedektor.sinyal_gurultu_orani` ile hesaplanabilir, bant başına gökyüzü parlaklıkları eklenmelidir.
- **Dedektör.** Kızılötesi diziler doğrusallıktan sapar ve yıkımsız okuma (rampa örnekleme) kullanır; doğrusallık düzeltmesi ve rampa uydurma ayrı fonksiyonlardır.
- **Sönümleme.** Yakın kızılötesi renk artığından (`H − K`) görsel sönümlemenin kestirimi; sönümleme yasası gerekir [CCM1989; Fitzpatrick1999].
- **Renk düzeltmesi.** WISE W3 ve W4 sıfır noktaları kırmızı kaynaklarda yüzde birkaç düzeltme ister [Wright2010].
- **Özgün veri.** SkyView kesitleri yeniden örneklenmiştir. Fotometri için 2MASS ve WISE atlas görüntüleri ya da katalogları IRSA'dan alınmalıdır; IRSA TAP istemcisi hazırdır.

## Optik tarafta yapılacaklar

- Kozmik ışın temizliği [vanDokkum2001], ayrıştırma (örtüşen kaynaklar), PSF fotometrisi [Stetson1987].
- Astrometrik çözüm: saptanan kaynakların Gaia ile eşlenmesi ve WCS uyumu; FITS başlığındaki WCS'in okunması [GreisenCalabretta2002; CalabrettaGreisen2002].
- Fotometrik kalibrasyon: alan yıldızlarının Gaia ya da 2MASS kadirleriyle sıfır noktası.
- Gözlem planlamasına presesyon, nutasyon, kırılma, Güneş ve Ay konumu, alacakaranlık sınırları.
- FITS uzantıları ve tabloları.

## Bekleyen kaynak kayıtları

Kaynak saptama [BertinArnouts1996], açıklık fotometrisi ve Gauss PSF modeli [Howell2006], CCD sinyal-gürültü denklemi [Howell2006; MerlineHowell1995] ve Gaia sütun tanımları [Vallenari2023] için kaynak metinleri açılmamıştır. Bu fonksiyonlar sentetik veriyle sınanmıştır; ilgili kitap ve makaleler okunduğunda kayıtlar güncellenecektir.
