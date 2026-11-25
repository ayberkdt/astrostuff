# Yol haritası

Aşamalar bağımlılık sırasındadır: her aşama yalnızca kendinden öncekilere dayanır ve bir kabul projesiyle kapanır. Süre verilmemiştir; ilerleme kabul ölçütleriyle ölçülür. Projeler [08_ornek_projeler](08_ornek_projeler/) altında tanımlıdır.

## Aşama 0: İskelet

- Git deposu, `pyproject.toml`, `src/kozmos` yerleşimi, `ruff`, `mypy`, `pytest`, `hypothesis`.
- `_kaynak.py`: `@kaynak` dekoratörü, kütük, kaynakça okuyucu.
- `tests/test_kaynak_kurali.py`: kaynak kuralının makine denetimi.
- `kaynakca/` klasörünün depo köküne taşınması; `dogrula.py` sürekli denetime bağlanır.
- Karar: tanımlayıcıların dili (Türkçe ASCII ya da İngilizce).

Kabul: kaynak göstermeyen örnek bir fonksiyon testi düşürür; gösteren geçirir.

## Aşama 1: Temel

- `temel/sabitler.py` [CODATA2022; Prsa2016], `temel/birimler.py`, `temel/aci.py`, `temel/vektor.py`.
- `temel/sayisal/`: kök bulma, integral, ADD çözücüler, enterpolasyon.
- `istatistik/`: dağılımlar, sağlam istatistik, doğrusal uyum, belirsizlik yayılımı.
- `io/tablo.py`.

Kabul: Hubble projesi Basamak 0 [Hubble1929].

## Aşama 2: Zaman ve koordinat

- `zaman/`: Jülyen tarihi, ölçek dönüşümleri, yıldız zamanı.
- `koordinat/`: küresel dönüşümler, Düzey 1 presesyon ve nutasyon, ufuk dönüşümü, kırılma.
- `efemeris/`: Kepler çözücü, elemanlar, Düzey 1 Güneş, Ay ve gezegenler.
- `gozlem/planlama.py`, `gozlem/hava_kutlesi.py`.

Kabul: P1 (gözlem gecesi planlayıcı). ERFA ile Düzey 1 doğruluk sınırları içinde uyum.

## Aşama 3: Kozmoloji ve çıkarım

- `kozmoloji/friedmann.py`, `kozmoloji/uzaklik.py`.
- `istatistik/mcmc.py`: Metropolis-Hastings, topluluk örnekleyici, tanılar.
- `arsiv/pantheon.py`, `arsiv/onbellek.py`.

Kabul: Hubble projesi Basamak 1-3. Bu aşama projenin ilk bilim sonucunu verir ve dedektör ya da görüntü işleme gerektirmediği için öne alınmıştır.

## Aşama 4: Işınım ve dedektör modeli

- `isinim/`: kara cisim, kadir sistemleri, filtreler, sentetik fotometri, sönümleme.
- `gozlem/dedektor.py`, `gozlem/sinyal_gurultu.py`; sentetik görüntü üreteci.
- `io/fits.py`: okuyucu ve yazıcı.

Kabul: Vega ve Güneş sentetik fotometrisi; sentetik CCD karesinin üretimi.

## Aşama 5: Görüntü indirgeme ve fotometri

- `indirgeme/`: kalibrasyon, kozmik ışın, arka plan, birleştirme.
- `fotometri/`: kaynak saptama, açıklık ve PSF fotometrisi, kalibrasyon.
- `astrometri/`: TAN izdüşümü, katalog eşleme, WCS uyumu.
- `arsiv/tap.py`, `arsiv/gaia.py`, `io/votable.py`.

Kabul: P2 (indirgeme hattı), P3 (açık küme), P4 (Gaia HR diyagramı).

## Aşama 6: Zaman serisi

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

## Aşama 9: Yıldız astrofiziği

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

1. Tanımlayıcı dili ve paket adı kararı.
2. Aşama 0 iskeleti.
3. [Hogg1999], [Brout2022], [Riess2022] ve Pantheon+SH0ES `README` dosyasının okunması; Aşama 3'ün denklem ve veri tanımlarının bu okumadan yazılması.
4. `temel/sabitler.py` için CODATA ve IAU değerlerinin kaynaktan aktarılması.
