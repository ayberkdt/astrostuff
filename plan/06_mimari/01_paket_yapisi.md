# Paket yapısı

Paketin çalışma adı `kozmos`'tur. Ad 25 Kasım 2026'da PyPI'da sorgulandı ve boş bulundu (`https://pypi.org/pypi/kozmos/json` → 404); `astrokit` ve `astrolab` adları alınmıştır. Ad bir öneridir, ilk kod yazılmadan değiştirilebilir.

## Tasarım ilkeleri

1. **Çekirdek saf fonksiyonlardan oluşur.** Bir hesap fonksiyonu girdisini değiştirmez, dosya okumaz, ağa çıkmaz, küresel duruma bakmaz. Girdi/çıktı ve ağ erişimi ayrı katmandadır (`kozmos.io`, `kozmos.arsiv`).
2. **Çalışma zamanı bağımlılığı yalnızca NumPy ve SciPy'dir.** Astropy, ERFA, Skyfield, photutils gibi kütüphaneler hesap yapmak için değil, test kâhini (oracle) olarak kullanılır ve `dogrulama` adlı isteğe bağlı bağımlılık grubunda durur. Amaç algoritmaları yeniden yazmaktır; `astropy`'yi sarmalayan bir fonksiyon bu amacı karşılamaz.
3. **Her açık fonksiyon bir kaynağa bağlıdır.** Kural ve denetimi [02_kod_standartlari_ve_kaynak_gosterme.md](02_kod_standartlari_ve_kaynak_gosterme.md) dosyasındadır.
4. **Birimler adlarda ve tiplerde görünür.** Çekirdek fonksiyonlar SI ya da açıkça adlandırılmış astronomi birimiyle `float`/`ndarray` alır; argüman adı birimi taşır (`uzaklik_pc`, `aci_rad`, `dalga_boyu_m`). Kullanıcıya dönük katmanda hafif bir `Nicelik` tipi birim dönüşümünü yapar. Gerekçe: birim nesnesi iç döngüde yavaştır, adsız `float` ise hata kaynağıdır; argüman adına gömülü birim ikisinin ortasıdır.
5. **Vektörleştirme varsayılandır.** Her fonksiyon skaler ve dizi girdisinde aynı sonucu verir; yayılım (broadcasting) kuralları NumPy'ninkilerdir.
6. **Katmanlar aşağıya doğru bağımlıdır.** Bir modül yalnızca kendisinden alt katmandaki modülü içe aktarır; döngüsel içe aktarma testle engellenir.

## Katmanlar

```
Katman 0  temel        sabitler, birimler, açı ve vektör yardımcıları, sayısal yöntemler
Katman 1  zaman        zaman ölçekleri, Jülyen tarihi, yıldız zamanı
          koordinat    küresel dönüşümler, presesyon-nutasyon, aberasyon, kırılma
Katman 2  efemeris     Kepler problemi, yörünge elemanları, gezegen konumları
          isinim       kara cisim, kadir sistemleri, filtreler, sönümleme
Katman 3  gozlem       planlama, hava kütlesi, S/N, dedektör modeli
          indirgeme    bias/dark/flat, kozmik ışın, arka plan
          fotometri    açıklık ve PSF fotometrisi, kalibrasyon
          astrometri   kaynak eşleme, WCS çözümü
          tayf         çıkarım, dalga boyu kalibrasyonu, çizgi ölçümü
          zamanserisi  periyodogram, katlama, geçiş araması
Katman 4  yildiz       yıldız yapısı, HR diyagramı, izokronlar
          gokada       potansiyeller, dönme eğrileri, ışınım profilleri
          kozmoloji    Friedmann denklemleri, uzaklıklar, Hubble diyagramı
          istatistik   olabilirlik, MCMC, uyum, belirsizlik yayılımı
Katman 5  arsiv        TAP/ADQL, katalog ve veri indirme (ağ)
          io           FITS, VOTable, tablo okuma-yazma (disk)
          projeler     uçtan uca analizler
```

`istatistik` Katman 4'te gösterilse de yalnızca Katman 0'a bağımlıdır ve her alan modülü onu kullanabilir.

## Dizin yerleşimi

```
Astronomy-and-Astrophysics/
├── plan/                     bu klasör
├── src/kozmos/
│   ├── temel/                sabitler.py, birimler.py, aci.py, vektor.py, sayisal/
│   ├── zaman/                julyen.py, olcekler.py, yildiz_zamani.py
│   ├── koordinat/            kuresel.py, cerceveler.py, presesyon.py, nutasyon.py,
│   │                         aberasyon.py, kirilma.py, paralaks.py
│   ├── efemeris/             kepler.py, elemanlar.py, gunes.py, ay.py, gezegen.py
│   ├── isinim/               kara_cisim.py, kadir.py, filtre.py, sonumleme.py
│   ├── gozlem/               planlama.py, hava_kutlesi.py, sinyal_gurultu.py, dedektor.py
│   ├── indirgeme/            kalibrasyon.py, kozmik_isin.py, arka_plan.py, birlestirme.py
│   ├── fotometri/            aciklik.py, psf.py, kaynak_bulma.py, kalibrasyon.py
│   ├── astrometri/           eslestirme.py, wcs.py, izdusum.py
│   ├── tayf/                 cikarim.py, dalgaboyu.py, cizgi.py, kirmiziya_kayma.py
│   ├── zamanserisi/          periyodogram.py, katlama.py, gecis.py
│   ├── yildiz/               yapi.py, hr.py, izokron.py, imf.py
│   ├── gokada/               potansiyel.py, donme_egrisi.py, profil.py
│   ├── kozmoloji/            friedmann.py, uzaklik.py, hubble_diyagrami.py, standart_mum.py
│   ├── istatistik/           olabilirlik.py, mcmc.py, uyum.py, saglam.py, yayilim.py
│   ├── arsiv/                tap.py, gaia.py, pantheon.py, onbellek.py
│   ├── io/                   fits.py, votable.py, tablo.py
│   └── _kaynak.py            kaynak gösterme altyapısı
├── tests/
│   ├── birim/                kaynağın çalışılmış örneklerine karşı
│   ├── kahin/                astropy/erfa/skyfield çıktısına karşı
│   └── ozellik/              hypothesis ile değişmezler
├── projeler/                 hubble_diyagrami/, hr_diyagrami/, gecis_egrisi/ ...
├── veri/                     ham/ (sürüm denetimi dışı), referans/ (küçük, sürüm denetiminde)
├── kaynakca/                 kaynaklar.bib ve doğrulama araçları 
└── pyproject.toml
```

Modül ve fonksiyon adları Türkçedir ve ASCII'ye indirgenmiştir (`ı`, `ş`, `ğ` kullanılmaz). İlk sürüm bu kuralla yazılmıştır.

## Veri tipleri

Çekirdek tipler değişmez veri sınıflarıdır (`@dataclass(frozen=True, slots=True)`):

- `Zaman`: iki `float64` ile tutulan Jülyen tarihi (`jd1`, `jd2`) ve ölçek etiketi (`"UTC"`, `"TT"`, `"TDB"` ...). İki parçalı gösterim, tek `float64`'ün J2000 yakınında yaklaşık 40 mikrosaniyede kalan çözünürlüğünü nanosaniye altına indirir; SOFA'nın kullandığı gösterim budur [SOFA2023].
- `GokKonumu`: çerçeve etiketi (`"ICRS"`, `"FK5"`, `"galaktik"`, `"ekliptik"`), iki açı (radyan), isteğe bağlı uzaklık, öz hareket ve dikine hız.
- `Gozlemci`: jeodezik enlem, boylam, yükseklik ve referans elipsoidi.
- `YorungeElemanlari`: `a, e, i, Ω, ω, M0`, devir ve merkez cismin `GM` değeri.
- `Kozmoloji`: `H0`, `Ωm`, `ΩΛ`, `Ωr`, `Ωk`, `w0`, `wa`.
- `Olcum`: değer, belirsizlik ve isteğe bağlı kovaryans göstergesi.

Görüntü, tayf ve ışık eğrisi için kaplar veriyi, maskeyi, belirsizliği ve üstveriyi (başlık, işlem geçmişi) birlikte taşır; her indirgeme adımı yeni bir kap döndürür ve geçmişe bir kayıt ekler.

## Bağımlılık grupları

- Çekirdek: `numpy`, `scipy`.
- `dogrulama`: `astropy`, `pyerfa`, `skyfield`, `photutils`, `sep`, `emcee` ve diğer kâhinler.
- `gelistirme`: `pytest`, `hypothesis`, `ruff`, `mypy`.
- `arsiv`: ağ erişimi için `httpx` ya da standart kütüphane; karar o modül yazılırken verilir.

Sürüm numaraları [../02_saha_arastirmasi](../02_saha_arastirmasi/) altındaki envanterde, okundukları tarihle kayıtlıdır.
