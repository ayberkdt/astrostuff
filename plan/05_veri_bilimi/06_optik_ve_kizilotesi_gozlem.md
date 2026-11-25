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

- **Gökyüzü çıkarma (yazıldı).** `indirgeme/gokyuzu.py` kaydırmalı karelerden durağan ve dinamik gökyüzü modeli kurar [Schirmer2013]. Sentetik dizide ölçülenler: kalabalık alanda nesne maskesi olmadan açıklık akısı %6,8 eksik çıkar, maskeyle %0,6; yığındaki en yüksek değeri atmak boş gökyüzünde artığı gürültünün onda biri kadar artıya kaydırır. Gerçek SOFI Ks dizisinde (16 ham kare) büyük ölçekli artık 151 ADU'dan 0,8 ADU'ya indi; kayıt `belgeler/kizilotesi_indirgeme.json`.
- **Isıl arka plan.** 3 µm'nin ötesinde teleskop ve atmosferin kendi ışıması baskındır; kısa poz, hızlı okuma ve nodlama gerekir. Arka plan foton gürültüsü `gozlem.dedektor.sinyal_gurultu_orani` ile hesaplanabilir, bant başına gökyüzü parlaklıkları eklenmelidir.
- **Dedektör.** Kızılötesi diziler doğrusallıktan sapar ve yıkımsız okuma (rampa örnekleme) kullanır; doğrusallık düzeltmesi ve rampa uydurma ayrı fonksiyonlardır.
- **Sönümleme.** Yakın kızılötesi renk artığından (`H − K`) görsel sönümlemenin kestirimi; sönümleme yasası gerekir [CCM1989; Fitzpatrick1999].
- **Renk düzeltmesi.** WISE W3 ve W4 sıfır noktaları kırmızı kaynaklarda yüzde birkaç düzeltme ister [Wright2010].
- **Özgün veri.** SkyView kesitleri yeniden örneklenmiştir. Fotometri için 2MASS ve WISE atlas görüntüleri ya da katalogları IRSA'dan alınmalıdır; IRSA TAP istemcisi hazırdır.

## Optik tarafta yapılacaklar

- PSF fotometrisi [Stetson1987].
- Bütün gökte arayan dizinli astrometrik çözücü. Yazılanlar: yaklaşık WCS'ten katalogla çözüm (`astrometri/cozum.py`) [CalabrettaGreisen2002] ve bölgenin kataloğu verildiğinde başlangıç WCS'i olmadan üçgen eşlemeyle çözüm (`astrometri/ucgen.py`) [Groth1986]. Kör çözüm doğrulama adımı taşır: gerçek SOFI yığınında birkaç yanlış çiftten kurulan çözüm yoğun katalogda rastlantıyla 17 yıldız eşleştirmiş, seyrek alanda ölçeği çökmüş bir çözüm bütün kaynakları tek yıldıza düşürmüştü.
- SOFI'nin maskeli düz alan kareleriyle sıfır düzeyi düzeltmesi; `.fits.Z` açma. Sadeleştirilmiş kubbe düz alanı uygulandı.
- Fotometrik kalibrasyon hattı. Sıfır noktası örnek betikte 2MASS kadirleriyle ölçüldü (40 yıldızda 0,021 kadir saçılım); kitaplık fonksiyonu değil.
- Gözlem planlamasına nutasyon, aberasyon, Güneş ve Ay konumu, alacakaranlık sınırları. Presesyon (`koordinat/presesyon.py`) ve kırılma (`gozlem/kirilma.py`) yazıldı [ERFA2023].
- FITS tabloları ve sıkıştırılmış görüntüler. Görüntü uzantıları yazıldı.

## Optik tarafta yazılanlar

- Kozmik ışın temizliği: `indirgeme/kozmik_isin.py`, Laplace kenar saptama [vanDokkum2001]. Sentetik karede tamlık %95'in üstünde, yanlış işaretleme binde ikinin altında (tests/birim/test_kozmik_isin.py).
- Kaynak ayrıştırma: çok eşikli ağaç ve eyer ölçütü, havzalar öncelik kuyruklu taşkınla [Photutils2026]. `scipy.ndimage.watershed_ift` eğim görüntüsü beklediği için parlaklık görüntüsünde eyeri aşıyor (164'e 4 piksellik bölünme gözlendi) ve kullanılmadı.
- Gerçek görüntüde ölçüm: 2MASS K kesitinde K = 7-11 aralığındaki 61 katalog yıldızının 59'u bulundu; ayrıştırma, parlak yapay yıldızların geri bulunma oranını %84'ten %96'ya çıkardı (belgeler/gercek_goruntu_dogrulamasi.json).

## Bekleyen kaynak kayıtları

Kaynak saptama [BertinArnouts1996], açıklık fotometrisi ve Gauss PSF modeli [Howell2006], CCD sinyal-gürültü denklemi [Howell2006; MerlineHowell1995] ve Gaia sütun tanımları [Vallenari2023] için kaynak metinleri açılmamıştır. Bu fonksiyonlar sentetik veriyle sınanmıştır; ilgili kitap ve makaleler okunduğunda kayıtlar güncellenecektir.
