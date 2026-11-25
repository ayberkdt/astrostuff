# Yol haritası

Aşamalar bağımlılık sırasındadır: her aşama yalnızca kendinden öncekilere dayanır ve bir kabul projesiyle kapanır. Süre verilmemiştir; ilerleme kabul ölçütleriyle ölçülür. Projeler [08_ornek_projeler](08_ornek_projeler/) altında tanımlıdır.

## Aşama 0: İskelet (tamamlandı)

- Git deposu, `pyproject.toml`, `src/kozmos` yerleşimi, `ruff`, `mypy`, `pytest`, `hypothesis`.
- `_kaynak.py`: `@kaynak` dekoratörü, kütük, kaynakça okuyucu.
- `tests/test_kaynak_kurali.py`: kaynak kuralının makine denetimi.
- `kaynakca/` klasörü depo kökündedir; `dogrula.py` sürekli denetime bağlanır.
- Karar: tanımlayıcıların dili (Türkçe ASCII ya da İngilizce).

Kabul: kaynak göstermeyen örnek bir fonksiyon testi düşürür; gösteren geçirir. Bu davranış geliştirme sırasında gözlenmiştir: kaynaksız bırakılan `tum_sabitler` fonksiyonu kural testini düşürmüş ve gerekçeli altyapı muafiyeti bu yüzden eklenmiştir.

## Aşama 1: Temel (kısmen)

Yazılanlar: `temel/sabitler.py`, `temel/aci.py`, `istatistik/uyum.py`, `istatistik/saglam.py`. Kalanlar: `temel/vektor.py`, `temel/sayisal/`, belirsizlik yayılımı, Hubble 1929 tablosu.

- `temel/sabitler.py` [CODATA2022; Prsa2016], `temel/birimler.py`, `temel/aci.py`, `temel/vektor.py`.
- `temel/sayisal/`: kök bulma, integral, ADD çözücüler, enterpolasyon.
- `istatistik/`: dağılımlar, sağlam istatistik, doğrusal uyum, belirsizlik yayılımı.
- `io/tablo.py`.

Kabul: Hubble projesi Basamak 0 [Hubble1929].

## Aşama 2: Zaman ve koordinat (kısmen)

Yazılanlar: `zaman/julyen.py` (takvim, ERA, GMST, eğiklik), `zaman/olcekler.py` (TAI − UTC), `koordinat/donusum.py` (galaktik, ufuk), `efemeris/kepler.py`. Kalanlar: TDB dönüşümü, presesyon ve nutasyon, kırılma, Güneş ve Ay konumları, planlayıcı.

- `zaman/`: Jülyen tarihi, ölçek dönüşümleri, yıldız zamanı.
- `koordinat/`: küresel dönüşümler, Düzey 1 presesyon ve nutasyon, ufuk dönüşümü, kırılma.
- `efemeris/`: Kepler çözücü, elemanlar, Düzey 1 Güneş, Ay ve gezegenler.
- `gozlem/planlama.py`, `gozlem/hava_kutlesi.py`.

Kabul: P1 (gözlem gecesi planlayıcı). ERFA ile Düzey 1 doğruluk sınırları içinde uyum.

## Aşama 3: Kozmoloji ve çıkarım (tamamlandı)

Yazılanlar: `kozmoloji/uzaklik.py`, `kozmoloji/hubble_diyagrami.py`, `istatistik/mcmc.py`, `arsiv/pantheon.py`, `projeler/hubble/hubble_analizi.py`. Analiz Ωm = 0,332 ± 0,018 ve H0 = 73,53 ± 1,00 km s⁻¹ Mpc⁻¹ vermiştir; yayımlanan değerler 0,334 ± 0,018 ve 73,6 ± 1,1'dir [Brout2022]. Kayıt: `projeler/hubble/sonuc.json`.

- `kozmoloji/friedmann.py`, `kozmoloji/uzaklik.py`.
- `istatistik/mcmc.py`: Metropolis-Hastings, topluluk örnekleyici, tanılar.
- `arsiv/pantheon.py`, `arsiv/onbellek.py`.

Kabul: Hubble projesi Basamak 1-3. Bu aşama projenin ilk bilim sonucunu verir ve dedektör ya da görüntü işleme gerektirmediği için öne alınmıştır.

## Aşama 4: Işınım ve dedektör modeli (başlandı)

Yazılanlar: `isinim/kadir.py`, `isinim/kizilotesi.py` (2MASS ve WISE sıfır noktaları), `gozlem/dedektor.py` (sinyal modeli, sentetik ham kare, yıldız alanı, CCD denklemi), `gozlem/planlama.py` (görünürlük, meridyen geçişi), `io/fits.py` (birincil görüntü birimi). Kalanlar: filtreler ve sentetik fotometri, sönümleme, FITS uzantıları.

- `isinim/`: kara cisim, kadir sistemleri, filtreler, sentetik fotometri, sönümleme.
- `gozlem/dedektor.py`, `gozlem/sinyal_gurultu.py`; sentetik görüntü üreteci.
- `io/fits.py`: okuyucu ve yazıcı.

Kabul: Vega ve Güneş sentetik fotometrisi; sentetik CCD karesinin üretimi.

## Aşama 5: Görüntü indirgeme ve fotometri (başlandı)

Yazılanlar: `indirgeme/kalibrasyon.py` (kare birleştirme, bias/karanlık/düz alan düzeltmesi, foton aktarımı), `fotometri/kaynak.py` (arka plan haritası, kaynak saptama, açıklık fotometrisi), `arsiv/tap.py`, `arsiv/gaia.py`, `arsiv/skyview.py`, `gorsel/germe.py`. Gaia HR diyagramı (P4) ile optik ve kızılötesi karşılaştırması üretildi. Kalanlar: kozmik ışın, ayrıştırma, PSF fotometrisi, astrometri, kalibrasyon. Ayrıntı: [05_veri_bilimi/06_optik_ve_kizilotesi_gozlem.md](05_veri_bilimi/06_optik_ve_kizilotesi_gozlem.md).

- `indirgeme/`: kalibrasyon, kozmik ışın, arka plan, birleştirme.
- `fotometri/`: kaynak saptama, açıklık ve PSF fotometrisi, kalibrasyon.
- `astrometri/`: TAN izdüşümü, katalog eşleme, WCS uyumu.
- `arsiv/tap.py`, `arsiv/gaia.py`, `io/votable.py`.

Kabul: P2 (indirgeme hattı), P3 (açık küme), P4 (Gaia HR diyagramı).

## Aşama 6: Zaman serisi (başlandı)

Yazılanlar: `zamanserisi/periyodogram.py` (klasik ve yüzen ortalamalı Lomb-Scargle).

- `zamanserisi/`: periyodogram, katlama, eğilim giderme, BLS ve TLS.
- Geçiş modeli; barisentrik düzeltme (`zaman` modülünün Düzey 2'ye çıkarılması).
- MAST erişimi.

Kabul: P5 (ötegezegen geçişi), P6 (Sefeid dönem-ışınım gücü bağıntısı).

## Aşama 7: Tayf

- `tayf/`: iz bulma, en iyi çıkarım, dalga boyu ve akı kalibrasyonu, çizgi ölçümü, kırmızıya kayma.
- SDSS erişimi.

Kabul: P7 (kırmızıya kayma ölçümü).

## Aşama 8: Yüksek doğruluk ve dinamik

- `koordinat/` Düzey 2 ve 3: IAU 2006/2000A, tam indirgeme zinciri.
- `efemeris/`: DE440 okuyucu, N cisim integratörleri, yörünge belirleme.
- `gokada/`: potansiyeller, yörüngeler, dönme eğrileri.

Kabul: P8, P9, P10. ERFA ile `1e-10` rad uyum.

## Aşama 9: Yıldız astrofiziği (başlandı)

Yazılanlar: `yildiz/` (politroplar, ölçekleme bağıntıları, Chandrasekhar kütlesi, başlangıç kütle fonksiyonları), `yildizolusumu/` (Jeans, Bonnor-Ebert, viryal parametre, Kennicutt-Schmidt, Toomre Q) ve `karadelik/` (Schwarzschild ve Kerr yarıçapları, yığılma, Hawking ışınımı, birleşen çiftler). Ayrıntı: [04_fizik/08_kara_delik_yildiz_ve_yildiz_olusumu.md](04_fizik/08_kara_delik_yildiz_ve_yildiz_olusumu.md).

- `yildiz/`: politroplar, basit yapı modeli, izokron okuma ve uydurma, kütle fonksiyonu.

Kabul: P11; P3'ün izokron uyumunun olabilirlik tabanlı sürümü.

## Aşama 10: İleri analizler

- Hubble projesi Basamak 4-5: seçim etkileri, birleşik sondalar.
- İç içe örnekleme, hiyerarşik modeller, Gauss süreçleri.
- Görüntü farkı ve geçici kaynaklar; çekimsel dalga; büyük ölçekli yapı.

Her konu başlamadan önce kendi saha taraması yapılır.

## Aşamalar arası kurallar

- Bir aşama, kabul projesi yayımlanmış sonucu yeniden üretmeden kapanmaz.
- Her modül dört sınamadan geçer ([06_mimari/03_test_ve_dogrulama.md](06_mimari/03_test_ve_dogrulama.md)).
- Bir fonksiyon yazılmadan önce atıf verilen kaynağın ilgili bölümü okunur ([02_saha_arastirmasi/04_arastirmanin_sinirlari.md](02_saha_arastirmasi/04_arastirmanin_sinirlari.md)).
- Yeni kaynak önce `kaynaklar.bib` dosyasına eklenir ve `dogrula.py` ile sınanır.
- Kapsam büyümesi ileri aşama listesine yazılır; yürüyen aşamaya eklenmez.

## İlk adımlar

İlk dört adım (tanımlayıcı dili, iskelet, kaynakların okunması, sabitlerin aktarılması) tamamlanmıştır. Sıradakiler:

1. `bekliyor` durumundaki 16 kaynak kaydının kapatılması ([DURUM.md](DURUM.md) listesi): CCD sinyal-gürültü denklemi, Gauss PSF, kaynak saptama, açıklık fotometrisi, Kerr ISCO enerjisi, buharlaşma süresi, Salpeter eğimi. Kepler, Planck, sağlam kestirici ve indirgeme kayıtları açık kaynaklardan okunarak kapatılmıştır.
2. Hubble 1929 tablosunun makaleden aktarılması (Basamak 0).
3. Aşama 1'in kalanı: sayısal yöntemler, sağlam istatistik, belirsizlik yayılımı.
4. Aşama 2'nin kalanı: UTC ve TDB, presesyon ve nutasyon, gözlem planlayıcı (P1).
5. Hubble projesinde yayımlanan ve bulunan `H0` belirsizliği arasındaki farkın (1,1 ve 1,0) incelenmesi.
