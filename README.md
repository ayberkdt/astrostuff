# astrostuff

`kozmos`: gökbilim ve astrofizik hesapları için sıfırdan yazılan bir Python kitaplığı. Tek bir kuralı var:

> **Her fonksiyon, uyguladığı denklemin yayımlanmış kaynağını gösterir ve bağımsız bir hesaba karşı sınanır. Kaynağı olmayan fonksiyon testten geçemez.**

Kitaplık çalışırken yalnızca NumPy ve SciPy kullanır. Astropy ve ERFA hesap için değil, sonuçları karşılaştırmak için vardır.

- 223 fonksiyon, 35 sabit, 1056 test
- 262 kaynak kaydının 254'ünde denklem kaynağın metninden okundu; 8'i açıkça "bekliyor" işaretli
- Güncel sayılar: [plan/DURUM.md](plan/DURUM.md) (koddan üretilir)

## Kitaplıkla üretilen sonuçlar

Aşağıdaki şekillerin hepsi bu depodaki kodla, açık veriden ya da doğrudan fonksiyon çıktısından üretilmiştir. Üreten betikler [ornekler/](ornekler/) ve [projeler/](projeler/) klasörlerindedir.

### Hubble sabiti: Pantheon+ süpernovaları

![Pantheon+ Hubble diyagramı: 1580 Tür Ia süpernovanın uzaklık modülü, en iyi uyum eğrisi ve artıklar](belgeler/hubble_diyagrami.png)

Kod açık [Pantheon+SH0ES verisini](https://github.com/PantheonPlusSH0ES/DataRelease) indirir, 1701 × 1701 kovaryans matrisiyle olabilirliği kurar ve kendi yazdığımız MCMC örnekleyicisiyle parametreleri kestirir. Bulunan değerler yayımlananları yeniden üretir:

- Madde yoğunluğu: **Ωm = 0,332 ± 0,018** (yayımlanan 0,334 ± 0,018)
- Hubble sabiti: **H0 = 73,53 ± 1,00 km s⁻¹ Mpc⁻¹** (yayımlanan 73,6 ± 1,1)

Yayımlanan değerler Brout ve ark. (2022) makalesinin arXiv kaynağından okundu. Kayıt: [projeler/hubble/sonuc.json](projeler/hubble/sonuc.json).

Bulunan H0 belirsizliği yayımlanandan küçüktür. Zincirden bağımsız kapalı biçimli bir çözüm ([h0_belirsizlik_denetimi.py](projeler/hubble/h0_belirsizlik_denetimi.py)) aynı veriden 73,52 ± 1,02 verir ve zinciri doğrular; kırmızıya kayma kesimini değiştirmek belirsizliği en çok 1,06'ya çıkarır. Riess ve ark. (2022) temel sonucu da ± 1,01'dir. Yayımlanan ± 1,1 ile kalan fark denenen seçeneklerle üretilemedi; yayımlanan değer tek ondalıkla verildiği için farkın büyüklüğü 0,03 ile 0,13 arasındadır.

### Yıldız kataloğu: Gaia ile Güneş çevresi

![Gaia DR3 Hertzsprung-Russell diyagramı: 50 parsek içindeki 31 298 yıldız; anakol, beyaz cüce dizisi ve dev kolu](belgeler/hr_diyagrami.png)

Güneş'e 50 parsekten yakın 31 298 yıldız anakol, beyaz cüce dizisi ve dev kolunda toplanıyor. Katalog Gaia DR3'ten kendi yazdığımız TAP istemcisiyle indirilir (paralaks sinyal-gürültü oranı > 20, RUWE < 1,4). Gaia çok parlak yıldızlarda doyduğu için en parlak yıldızlar örneklemde yoktur.

### Aynı alan, üç dalga boyu: optik ve kızılötesi

![Alev Bulutsusu'nun optik, yakın kızılötesi ve orta kızılötesi görüntüleri; saptanan kaynak sayıları 17, 132 ve 15](belgeler/optik_kizilotesi.png)

Alev Bulutsusu'nda (NGC 2024) toz, görünür ışıkta yıldızları gizler; 2 µm'de aynı alanda saptanan kaynak sayısı 17'den 132'ye çıkar. 12 µm'de ısınmış tozun kendi ışıması görünür; ortadaki kiremit rengi bölgede 10 247 piksel veri taşımaz (FITS dosyasında NaN). Bölge en parlak ışımanın ortasındadır; veri yokluğunun nedeni (dedektör doyması ya da maskeleme) tarama belgelerinden doğrulanmadı. Görüntüler NASA SkyView'dan indirilir, kendi FITS okuyucumuzla okunur; arka plan haritası ve kaynak saptama `kozmos.fotometri` ile yapılır.

<img src="belgeler/alev_bulutsusu_jhk.png" width="360" alt="Alev Bulutsusu'nun 2MASS J, H ve K bantlarından renkli bileşimi; toz şeridinin içinde kızarmış genç yıldız kümesi">

Aynı alanın 2MASS J, H ve K bantlarından renk koruyan bileşimi (Lupton ve ark. 2004 germesi). Toz şeridinin içindeki kırmızı yıldızlar K bandında J bandına göre çok daha parlaktır; bu, tozun içinden gelen ışığın kızarmasıyla uyumludur.

### Gerçek görüntüde doğrulama: 2MASS kataloğuna karşı

![Solda K kadirine göre saptanan kaynak kesri, katalog yıldızları ve yapay yıldızlar için; sağda saptanan konumların katalog konumlarından farkı](belgeler/gercek_goruntu_dogrulamasi.png)

Kaynak saptama, WCS dönüşümü ve katalog eşleme aynı alanın 2MASS K görüntüsünde (1 yay saniyesi/piksel) 2MASS Nokta Kaynak Kataloğu'na karşı sınandı ([gercek_goruntu_dogrulamasi.py](ornekler/gercek_goruntu_dogrulamasi.py), kayıt: [gercek_goruntu_dogrulamasi.json](belgeler/gercek_goruntu_dogrulamasi.json)):

- K = 7-11 aralığındaki 61 katalog yıldızının 59'u bulundu; görüntüye eklenen yapay yıldızların K ≤ 12 için en az %94'ü, K = 14'te %51'i geri bulundu.
- Saptanan 281 kaynağın %85'inin 2 yay saniyesi içinde katalog karşılığı var; kalanların bir kısmı bulutsunun parlak düğümleridir.
- Konum farkının ortancası sağ açıklıkta 0,16, dik açıklıkta 0,03 yay saniyesi; saçılım iki eksende 0,14 ve 0,13 yay saniyesi. Artıklar iki öbekte toplanıyor; nedeni incelenmedi (SkyView'ın yeniden örneklemesi olası).
- Açıklık kadiri ile katalog K kadiri arasındaki sıfır noktasının saçılımı 40 yalıtılmış yıldızda 0,021 kadir.

Örtüşen kaynakları ayıran adım bu sınamada eklendi: ayrıştırma olmadan parlak yapay yıldızların %84'ü bulunuyordu. Katalog aynı taramanın verisinden üretildiği için bu sınama kodu ölçer, 2MASS'in kendisini değil.

### Parlak yıldızlar: gökyüzü haritası

![Parlak Yıldız Kataloğu'ndaki 9096 yıldızın galaktik koordinatlarda gökyüzü haritası; Samanyolu düzlemi boyunca yoğunlaşma](belgeler/gokyuzu_haritasi.png)

Parlak Yıldız Kataloğu'nun (5. baskı) konum ve kadir taşıyan 9096 yıldızı, galaktik koordinatlarda. Yıldızların %47'si galaktik enlemin ±20° kuşağındadır; düzgün dağılımda bu oran %34 olurdu. Gaia'nın doyduğu çıplak gözle görülen yıldızlar bu katalogdan gelir.

### Kara delikler, yıldızlar, yıldız oluşumu

![Dönen kara delikte en içteki kararlı yörüngenin yarıçapı ve yığılma verimi, dönme parametresine karşı](belgeler/kara_delik_donme.png)

Kara delik döndükçe en içteki kararlı yörünge 6'dan 1,24 çekim yarıçapına iner ve yığılan maddenin ışınıma çevrilen kesri %5,7'den %32'ye çıkar (a* = 0,998).

<p>
<img src="belgeler/lane_emden.png" width="49%" alt="Lane-Emden denkleminin n = 0, 1, 1,5 ve 3 politrop indisleri için çözümleri">
<img src="belgeler/kutle_fonksiyonlari.png" width="49%" alt="Salpeter, Kroupa ve Chabrier başlangıç kütle fonksiyonları">
</p>

Solda politrop yıldız modelleri: indis büyüdükçe yıldız merkezde yoğunlaşır (n = 3 için merkez yoğunluğu ortalamanın 54 katı). Sağda üç başlangıç kütle fonksiyonu, 0,1-50 Güneş kütlesi aralığında aynı toplam sayıya normalleştirilmiş: 1 Güneş kütlesinin üstünde eğimleri aynıdır, altında Salpeter çok daha fazla küçük yıldız öngörür.

<img src="belgeler/jeans_kutlesi.png" width="62%" alt="Jeans kütlesinin gaz yoğunluğuna bağlılığı, 10, 20 ve 50 kelvin için">

Bir gaz bulutunun kendi çekimiyle çökebilmesi için gereken en küçük kütle. 10 K sıcaklıkta ve santimetreküpte 10⁴ parçacık yoğunluğunda yaklaşık 3 Güneş kütlesidir.

## Neler var

**Gözlem ve veri**

- `arsiv`: TAP istemcisi, Gaia DR3 yıldız örneklemleri, 2MASS nokta kaynak koni sorgusu, Parlak Yıldız Kataloğu, SkyView görüntü kesitleri (DSS2, 2MASS, WISE), Pantheon+ verisi; her indirme SHA-256 özetiyle kayda geçer
- `io`: FITS okuyucu ve yazıcı (FITS 4.0 standardından); birincil birim ve görüntü uzantıları
- `gozlem`: hedef görünürlüğü ve hava kütlesi (presesyon ve atmosfer kırılmasıyla), CCD sinyal modeli, sentetik ham kare, sinyal-gürültü oranı
- `indirgeme`: sigma kırpmalı kare birleştirme, bias/karanlık/düz alan düzeltmesi, kazanç ve okuma gürültüsü ölçümü, tek karede kozmik ışın temizliği (L.A.Cosmic), kaydırmalı kızılötesi karelerden gökyüzü çıkarma
- `fotometri`: arka plan haritası, kaynak saptama ve örtüşen kaynakların ayrıştırılması, açıklık fotometrisi
- `astrometri`: FITS WCS piksel-gök dönüşümü (TAN izdüşümü), katalog çapraz eşleme
- `gorsel`: asinh germe ve renk koruyan RGB bileşimi
- `isinim`: kadir sistemleri, Planck fonksiyonu, 2MASS ve WISE sıfır noktaları
- `zamanserisi`: Lomb-Scargle periyodogramı

**Astrofizik**

- `karadelik`: Schwarzschild ve Kerr yarıçapları, gölge boyutu, yığılma (Eddington, Bondi, ince disk), Hawking ışınımı, birleşen çiftlerin çekimsel dalga frekansı
- `yildiz`: Lane-Emden politropları, yıldız ölçekleme bağıntıları, Chandrasekhar kütlesi, başlangıç kütle fonksiyonları
- `yildizolusumu`: Jeans ve Bonnor-Ebert kütleleri, serbest düşme süresi, viryal parametre, Kennicutt-Schmidt yasası, Toomre Q
- `kozmoloji`: uzaklık ölçüleri, Hubble diyagramı olabilirliği
- `efemeris`: Kepler denklemi ve yörünge bağıntıları

**Altyapı**

- `temel`: CODATA 2022 sabitleri, IAU nominal değerleri, açı aritmetiği
- `zaman`, `koordinat`: Jülyen tarihi, artık saniyeler, yıldız zamanı, galaktik ve ufuk dönüşümleri, IAU 2006 presesyonu
- `istatistik`: doğrusal uyum, MCMC, yakınsama tanıları, sağlam kestiriciler

## Kaynak kuralı nasıl işler

```python
@kaynak("Hogg1999", yer="Denk. 15", teyit="okundu")
def esyuruyen_uzaklik_mpc(z, kozmoloji):
    """Bakış doğrultusundaki eşyürüyen uzaklığı hesaplar.

    Kaynaklar
    ---------
    [Hogg1999] Denk. 15.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji_kahin.py
    """
```

- `teyit="okundu"`: denklem kod yazılırken kaynağın metninden alındı (arXiv TeX dosyası, standart belgesi, kaynak kodu).
- `teyit="bekliyor"`: denklem bilinen biçimiyle yazıldı ve testlerle sınandı, ama kaynağın metni henüz açılmadı.

Bir test her açık fonksiyon için kaydın varlığını, anahtarın [kaynakçada](kaynakca/) tanımlı olduğunu, belge dizgisinin aynı kaynağı andığını ve gösterilen test dosyasının var olduğunu denetler. Kaynakçadaki künyeler de sınanır: [kaynakca/dogrula.py](kaynakca/dogrula.py) her DOI'yi Crossref'e, her arXiv kimliğini arXiv'e sorar.

Kaynak metnini açmak sonucu değiştirdi. Okuma sırasında bulunanlardan üçü:

- Bir derleme makalesindeki Kerr foton yörüngesi formülü, basıldığı biçimiyle ileri değil geri yönlü yörüngeyi veriyor.
- Jeans kütlesinin iki yaygın tanımı arasında 4π/3 kat fark var; fonksiyon ikisini de adıyla sunuyor.
- Bir kaynaktaki politrop tablosunun kütle katsayısı sütunu yalnızca n = 3 için standart tanımla uyuşuyor.

Ayrıntılar [plan/04_fizik/08_kara_delik_yildiz_ve_yildiz_olusumu.md](plan/04_fizik/08_kara_delik_yildiz_ve_yildiz_olusumu.md) dosyasındadır.

## Kurulum ve kullanım

Python 3.12 ya da üstü gerekir.

```
pip install -e ".[dogrulama,gelistirme,gorsel]"
pytest -m "not ag"
```

```python
import numpy as np
from kozmos.kozmoloji.uzaklik import Kozmoloji, isinim_gucu_uzakligi_mpc
from kozmos.karadelik.temel import isco_isinim_verimliligi
from kozmos import kaynakca_raporu

isinim_gucu_uzakligi_mpc(np.array([0.1, 1.0]), Kozmoloji.duz(h0=70.0, omega_m=0.3))  # [460.30, 6607.66] Mpc
isco_isinim_verimliligi(0.0)  # 0.0572: dönmeyen kara delikte yığılma verimi
print(kaynakca_raporu(isco_isinim_verimliligi))  # hangi yayına dayanıyor, okundu mu
```

Şekilleri yeniden üretmek için (veriyi kendileri indirir):

```
python projeler/hubble/hubble_analizi.py     # yaklaşık 12 dakika
python projeler/hubble/hubble_sekli.py
python ornekler/hr_diyagrami.py
python ornekler/optik_kizilotesi.py
python ornekler/fizik_galerisi.py
python ornekler/gokyuzu_haritasi.py
python ornekler/gercek_goruntu_dogrulamasi.py
```

Depodaki şekiller yerel bir figür denetim aracıyla (`datawiz`) üretildi; araç bu depoda yoktur. Araç kurulu değilse betikler aynı veriyi düz matplotlib ile çizer ([ornekler/_cizim.py](ornekler/_cizim.py)); görünüm depodaki şekillerden farklı olur.

## Sınırlar

- Yeni yazılan fizik modülleri kendi testleriyle, kapalı çözümlerle ve kaynaklardaki sayılarla sınandı. Bağımsız bir kitaplıkla karşılaştırma zaman, koordinat, kozmoloji, istatistik, FITS ve periyodogram modüllerinde var.
- Gözlem planlaması çerçeve yanlılığı, presesyon ve isteğe bağlı kırılma içerir; nutasyon, aberasyon ve UT1 − UTC farkı yoktur. 30 rastgele hedefte astropy ile yön farkı en çok 0,45 yay dakikasıdır (presesyonsuz hesapta 26 yay dakikası). Kırılma modeli yaklaşık 5° yüksekliğin altında geçersizdir.
- FITS okuyucu birincil birimi ve görüntü uzantılarını okur; tablo uzantıları listelenir ama çözülmez, sıkıştırılmış görüntüler okunmaz.
- Kaynak ayrıştırma aralarında belirgin eyer olan çiftleri ayırır: 2 FWHM aralıkta akı oranı 2'ye kadar çiftlerin %92-98'i, oran 3'te varsayılan ayarla %14'ü ayrışır. Açıklık fotometrisi dar açıklıkta yüzde birkaç eksik ölçer.
- WCS yalnızca TAN izdüşümünü çözer; bozulma terimleri (SIP) ve görüntüden astrometrik çözüm (yıldız deseninden WCS bulma) yoktur.
- Kozmik ışın temizliği ve gökyüzü çıkarma sentetik veriyle sınandı; gerçek ham karelerle sınanmadı.
- SkyView kesitleri yeniden örneklenmiş görüntülerdir; fotometrik kalibrasyon için özgün arşiv verisi gerekir.

## Plan

[plan/](plan/00_OKU_BENI.md) klasörü projenin tasarımını taşır: saha araştırması, matematik, fizik, veri bilimi, mimari, yol haritası ve işlev envanteri. Plan belgelerindeki denklemler planlama düzeyindedir; bir fonksiyon yazılırken denklem kaynağından alınır.

## Lisans

Telif Hakkı © 2026 Ayberk. Tüm hakları saklıdır; ayrıntı [LICENSE](LICENSE) dosyasındadır. Bazı fonksiyonların izlediği ERFA algoritmalarına ilişkin bildirim [UCUNCU_TARAF_BILDIRIMLERI.md](UCUNCU_TARAF_BILDIRIMLERI.md) dosyasındadır. Analizlerin kullandığı veri depoya konmaz; çalışma anında kaynağından indirilir. Şekillerin bir kısmı DSS2, 2MASS, WISE, Gaia, Parlak Yıldız Kataloğu ve Pantheon+ verisinden türetilmiştir; hak saklama bu şekilleri kapsamaz. DSS verisi içeren `optik_kizilotesi.png` CC BY-NC-SA 4.0 ile paylaşılır, Gaia verisi CC BY-NC 3.0 IGO lisanslıdır. Her taramanın koşulları ve istenen teşekkür metinleri aynı bildirim dosyasındadır.

This publication makes use of data products from the Two Micron All Sky Survey, which is a joint project of the University of Massachusetts and the Infrared Processing and Analysis Center/California Institute of Technology, funded by the National Aeronautics and Space Administration and the National Science Foundation. This publication makes use of data products from the Wide-field Infrared Survey Explorer, which is a joint project of the University of California, Los Angeles, and the Jet Propulsion Laboratory/California Institute of Technology, funded by the National Aeronautics and Space Administration. This research has made use of the NASA/IPAC Infrared Science Archive, which is funded by the National Aeronautics and Space Administration and operated by the California Institute of Technology.
