# astrostuff

Gökbilim ve astrofizik hesapları için sıfırdan yazılan bir Python kitaplığı (`kozmos`) ve onun planı. Kitaplığın kuralı: **her fonksiyon uyguladığı denklemin yayımlanmış kaynağını gösterir ve bağımsız bir uygulamaya karşı sınanır.** Kaynağı olmayan fonksiyon test kapısından geçemez.

![Pantheon+ Hubble diyagramı: 1580 Tür Ia süpernovanın uzaklık modülü kırmızıya kaymaya karşı, bu depodaki kodla bulunan düz ΛCDM eğrisi ve artıklar](belgeler/hubble_diyagrami.png)

*Pantheon+ süpernovaları bu depodaki kodla bulunan düz ΛCDM eğrisini izliyor. Üstte 1580 süpernovanın uzaklık modülü ve en iyi uyum eğrisi, altta artıklar ve kırmızıya kayma kutularındaki ortancaları. Artıkların ortancası +0,004 mag, sağlam saçılımı 0,13 mag. Uyum tam kovaryans matrisiyle yapılmıştır.*

## İlk sonuç: Pantheon+SH0ES verisinden Hubble sabiti

Depodaki kod, açık [Pantheon+SH0ES veri sürümünü](https://github.com/PantheonPlusSH0ES/DataRelease) indirir, 1701 × 1701 kovaryans matrisiyle olabilirliği kurar ve kendi yazdığımız topluluk örnekleyiciyle parametreleri kestirir. Sonuçlar yayımlanan değerleri yeniden üretir:

- **Madde yoğunluğu** (yalnızca süpernovalar, düz ΛCDM): Ωm = 0,332 ± 0,018. Yayımlanan: 0,334 ± 0,018.
- **Hubble sabiti** (Sefeid kalibratörleriyle): H0 = 73,53 ± 1,00 km s⁻¹ Mpc⁻¹. Yayımlanan: 73,6 ± 1,1.

Yayımlanan değerler Brout ve ark. (2022, ApJ 938, 110) makalesinin arXiv kaynağından okunmuştur. Zincir 32 yürüyücü ve 2000 adımdır; bölünmüş R̂ en çok 1,03, etkin örnek sayısı yaklaşık 1600'dür. Çalıştırma kaydı (veri özetleri, parametreler, tanılar) [projeler/hubble/sonuc.json](projeler/hubble/sonuc.json) dosyasındadır.

Bulunan H0 belirsizliği yayımlanandan küçüktür (1,00 ve 1,1); nedeni henüz incelenmemiştir. Model uzaklığı tek kırmızıya kaymayla (`zHD`) hesaplanır ve alt eşik z > 0,01 alınmıştır.

```
python projeler/hubble/hubble_analizi.py     # veriyi indirir, yaklaşık 13 dakika
python projeler/hubble/hubble_sekli.py       # yukarıdaki şekil
```

## Kaynak gösterme

Her açık fonksiyon bir `@kaynak` kaydı taşır: kaynakçadaki anahtar, kaynağın içindeki yer ve kaydın kanıt statüsü.

```python
@kaynak("Hogg1999", yer="Denk. 15", teyit="okundu")
def esyuruyen_uzaklik_mpc(z, kozmoloji):
    """Bakış doğrultusundaki eşyürüyen uzaklığı hesaplar.

    D_C = D_H ∫₀ᶻ dz′ / E(z′)

    Kaynaklar
    ---------
    [Hogg1999] Denk. 15.

    Doğrulama
    ---------
    tests/kahin/test_kozmoloji_kahin.py
    """
```

- `teyit="okundu"`: denklem kod yazılırken kaynağın metninden alındı (arXiv TeX dosyası, NIST sabit tablosu, ERFA kaynak kodu).
- `teyit="bekliyor"`: denklem hatırlanan biçimiyle yazıldı ve testlerle sınandı, ama kaynağın metni henüz açılmadı.

`tests/test_kaynak_kurali.py` her açık fonksiyon için dört şeyi denetler: kayıt var mı, anahtar [kaynakçada](kaynakca/kaynaklar.bib) tanımlı mı, belge dizgisi aynı anahtarı anıyor mu, `Doğrulama` bölümündeki test dosyası gerçekten var mı. Kaynakçadaki künyeler de ayrıca sınanır: [kaynakca/dogrula.py](kaynakca/dogrula.py) her DOI'yi Crossref'e, her arXiv kimliğini arXiv'e, her ISBN'i Open Library'ye sorar ve başlık ile yılı karşılaştırır.

Güncel sayılar [plan/DURUM.md](plan/DURUM.md) dosyasındadır; bu dosya elle yazılmaz, kütükten üretilir. İlk sürümde 80 fonksiyon ve 35 sabit vardır; 83 kaynak kaydının 61 tanesi kaynağından okunmuş, 22 tanesi beklemektedir.

## Neler var

- **`temel`**: CODATA 2022 sabitleri, IAU 2015 nominal Güneş ve gezegen değerleri, açı aritmetiği, küre üzerinde ayrıklık.
- **`zaman`**: Gregoryen takvim ↔ Jülyen tarihi, TAI ↔ TT, artık saniye tablosuyla TAI − UTC, Dünya dönme açısı, Greenwich ve yerel ortalama yıldız zamanı, ekliptiğin ortalama eğikliği (IAU 2006).
- **`koordinat`**: ICRS ↔ galaktik, saat açısı ↔ azimut ve yükseklik.
- **`efemeris`**: Kepler denklemi, gerçek anomali, yörünge dönemi, vis-viva.
- **`isinim`**: bolometrik kadir (IAU 2015 B2), uzaklık modülü, Planck fonksiyonu.
- **`kozmoloji`**: `E(z)`, eşyürüyen, açısal çap ve ışınım gücü uzaklıkları, geriye bakış zamanı, eşyürüyen hacim, Doppler kayması ve özel hız; süpernova Hubble diyagramı olabilirliği.
- **`gozlem`** ve **`indirgeme`**: CCD sinyal modeli, sentetik ham kare ve yıldız alanı üreteci, CCD denklemiyle sinyal-gürültü oranı; sigma kırpmalı kare birleştirme, bias, karanlık ve düz alan düzeltmesi (varyans düzlemiyle), foton aktarım yöntemiyle kazanç ve okuma gürültüsü ölçümü.
- **`istatistik`**: sigma kırpma, MAD ve çift ağırlıklı kestiriciler, kovaryanslı doğrusal en küçük kareler, afin değişmez topluluk örnekleyici (gerdirme hamlesi), öz ilinti süresi, bölünmüş ve sıra normalleştirilmiş R̂.
- **`zamanserisi`**: klasik ve yüzen ortalamalı, ağırlıklı Lomb-Scargle periyodogramı.
- **`arsiv`**: Pantheon+SH0ES verisinin indirilmesi, SHA-256 özet kaydı, tablo ve kovaryans okuma.

Kitaplık çalışırken yalnızca NumPy ve SciPy kullanır. Astropy ve ERFA hesap için değil, sonuçları karşılaştırmak için vardır: uzaklıklar `astropy.cosmology` ile 10⁻⁹, zaman ve koordinat fonksiyonları ERFA ile 10⁻¹² rad, periyodogram `astropy.timeseries` ile 10⁻¹⁰ düzeyinde uyuşur. İndirgeme hattının kâhini sentetik veridir: bilinen bir sahneden üretilen ham kareler indirgenir ve sonuç girilen sahneyle karşılaştırılır; artıklar yansızdır ve varyans düzlemi gerçek saçılımı öngörür (χ²/N ≈ 1).

## Kurulum ve test

Python 3.12 ya da üstü gerekir.

```
pip install -e ".[dogrulama,gelistirme]"
pytest -m "not ag"
```

Testler üç türdür: kaynağın sayısal örneklerine ve kapalı çözümlere karşı (`tests/birim`), Astropy ve ERFA'ya karşı (`tests/kahin`), `hypothesis` ile değişmezlere karşı (`tests/ozellik`). Ağ gerektiren testler `-m ag` ile ayrıca koşulur.

```python
import numpy as np
from kozmos.kozmoloji.uzaklik import Kozmoloji, isinim_gucu_uzakligi_mpc
from kozmos import kaynakca_raporu

kozmoloji = Kozmoloji.duz(h0=70.0, omega_m=0.3)
isinim_gucu_uzakligi_mpc(np.array([0.1, 1.0]), kozmoloji)   # [460.30, 6607.66] Mpc
print(kaynakca_raporu(isinim_gucu_uzakligi_mpc))            # hangi yayına dayanıyor
```

## Plan

[plan/](plan/00_OKU_BENI.md) klasörü projenin tasarımını ve yol haritasını taşır:

- [Saha araştırması](plan/02_saha_arastirmasi/): var olan kütüphaneler, veri setleri, arşivler ve araştırmanın sınırları.
- [Matematik](plan/03_matematik/): küresel astronomi, zaman ölçekleri, sayısal yöntemler, istatistik, görüntü ve sinyal işleme.
- [Fizik](plan/04_fizik/): ışınım, yörünge mekaniği, yıldızlar, gökadalar, kozmoloji, dedektör ve atmosfer, görelilik ve yüksek enerji.
- [Veri bilimi](plan/05_veri_bilimi/): veri erişimi, görüntü indirgeme, fotometri, astrometri, tayf, zaman serisi, tekrarlanabilirlik.
- [Mimari](plan/06_mimari/), [yol haritası](plan/07_yol_haritasi.md), [örnek projeler](plan/08_ornek_projeler/) ve [işlev envanteri](plan/09_islev_envanteri.md).

Plan belgelerindeki denklemler planlama düzeyindedir ve hatırlanan bilgiye dayanır; bir fonksiyon yazılırken denklem kaynağından alınır. Hangi bilginin nasıl elde edildiği [plan/02_saha_arastirmasi/04_arastirmanin_sinirlari.md](plan/02_saha_arastirmasi/04_arastirmanin_sinirlari.md) dosyasında yazılıdır.

## Sırada ne var

Yer tabanlı gözlem hattının devamı: zaman ölçeklerinin tamamlanması, presesyon ve nutasyon, gözlem planlayıcı; kozmik ışın temizliği, arka plan haritası, kaynak saptama, açıklık fotometrisi ve astrometri. Dedektör ve indirgeme fonksiyonlarının kaynak kayıtları (Howell, Janesick) henüz `bekliyor` durumundadır; kitaplar açılıp denklemler teyit edilecektir. Ayrıntı [yol haritasında](plan/07_yol_haritasi.md).

## Lisans

Telif Hakkı © 2026 Ayberk. Tüm hakları saklıdır; ayrıntı [LICENSE](LICENSE) dosyasındadır. Bazı fonksiyonların izlediği ERFA algoritmalarına ilişkin bildirim [UCUNCU_TARAF_BILDIRIMLERI.md](UCUNCU_TARAF_BILDIRIMLERI.md) dosyasındadır. Analizlerin kullandığı veri depoya konmaz; çalışma anında kaynağından indirilir.
