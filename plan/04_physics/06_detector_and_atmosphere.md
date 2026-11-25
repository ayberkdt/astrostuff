# Dedektör, teleskop ve atmosfer fiziği

Hedef modüller: `gozlem/`, `indirgeme/` (model tarafı).

Yer tabanlı gözlemde foton kaynaktan çıkıp sayıya dönüşene kadar atmosferden, optikten ve dedektörden geçer. Veri indirgeme bu zincirin tersine çevrilmesidir; zincir modellenmeden indirgeme adımlarının neyi düzelttiği anlaşılmaz. Denklemler kodlanmadan önce anılan kaynaktan okunur.

## 1. Atmosfer

- **Hava kütlesi:** düzlem-paralel yaklaşım `X = sec z`; küresel atmosfer için yaklaşım formülleri [KastenYoung1989; Young1994]. `z > 80°` için formüller ayrışır; her birinin geçerlilik aralığı belge dizgisinde yazılır.
- **Sönümleme:** Bouguer yasası `m_gözlenen = m_0 + k_λ X`. `k_λ` Rayleigh saçılması (`∝ λ⁻⁴`), aerosol ve ozon bileşenlerinden oluşur; gece içinde standart yıldızlardan ölçülür. İkinci mertebe renk terimi geniş bantlarda gerekir.
- **Kırılma ve dağılma:** [../03_matematik/01_kuresel_astronomi_ve_donusumler.md](../03_matematik/01_kuresel_astronomi_ve_donusumler.md).
- **Görüş (seeing):** Kolmogorov türbülansı, Fried parametresi `r0 ∝ λ^(6/5)`, görüş diski `FWHM ≈ 0,98 λ/r0`; `X^0,6` ile hava kütlesi bağımlılığı [Fried1966].
- **Gökyüzü parlaklığı:** hava ışıması, burçlar ışığı, Ay'ın katkısı (evre ve açısal uzaklık), ışık kirliliği; Ay parlaklığı modeli [KrisciunasSchaefer1991].
- **Tellürik soğurma:** O₂ ve H₂O bantları; tayfta düzeltme.

## 2. Teleskop ve optik

- Toplama alanı, odak oranı, plaka ölçeği: `ölçek [″/mm] = 206265 / f [mm]`; piksel ölçeği.
- Kırınım: Airy deseni, `θ = 1,22 λ/D`; merkezi engel etkisi.
- Nokta dağılım fonksiyonu modelleri: Gauss, Moffat, Airy; görüş + optik + izleme bileşimi [King1971; Moffat1969].
- Örnekleme: Nyquist ölçütü `FWHM ≳ 2 piksel`; eksik örneklemenin fotometri ve astrometriye etkisi.
- Geçirgenlik zinciri: atmosfer × aynalar × filtre × dedektör kuantum verimi; sistem sıfır noktası bu zincirden öngörülür ve ölçülenle karşılaştırılır.

## 3. Dedektör

CCD ve CMOS dedektörün sinyal modeli [Howell2006, Böl. 3-4; Janesick2007]:

```
ham[ADU] = ( akı · t · QE · düz_alan + karanlık_akım · t ) / kazanç + bias + gürültü
```

- **Kazanç** [e⁻/ADU], **okuma gürültüsü** [e⁻], **karanlık akım** [e⁻/s] ve sıcaklık bağımlılığı.
- **Bias** düzeyi ve yapısı; aşırı tarama (overscan) bölgesi.
- **Düz alan:** pikselden piksele duyarlılık, vinyet, toz gölgeleri; dalga boyuna bağlıdır.
- **Doğrusallık ve doyma:** tam kuyu sığası, doğrusallıktan sapma eğrisi, taşma (blooming).
- **Foton aktarım eğrisi:** varyans-ortalama doğrusunun eğiminden kazanç, kesişiminden okuma gürültüsü [Janesick2007].
- Kusurlar: sıcak ve ölü pikseller, kötü sütunlar, yük aktarım verimsizliği, saçak (fringing), kalıntı görüntü.
- CMOS farkları: piksel başına kazanç ve gürültü, sabit desen gürültüsü, yuvarlanan perde, amfi parlaması.
- Kozmik ışınlar: keskin kenarlı, PSF'ten dar izler [vanDokkum2001].

## 4. Sinyal-gürültü oranı

CCD denklemi [Howell2006, Böl. 4; MerlineHowell1995]:

```
S/N = N* / sqrt( N* + n_pix (1 + n_pix/n_B) (N_S + N_D + N_R² + G² σ_f²) )
```

`N*` kaynak elektronları, `n_pix` açıklıktaki piksel sayısı, `n_B` arka plan kestiriminde kullanılan piksel sayısı, `N_S` piksel başına gökyüzü, `N_D` karanlık akım, `N_R` okuma gürültüsü, `G² σ_f²` sayısallaştırma gürültüsü.

- Rejimler: kaynak baskın (`S/N ∝ √t`), gökyüzü baskın, okuma gürültüsü baskın (`S/N ∝ t`).
- En iyi açıklık yarıçapı: Gauss PSF ve gökyüzü baskın durumda yaklaşık `0,68 FWHM`; türetimi projede yazılır.
- Poz süresi hesaplayıcı: hedef kadir ve S/N için süre; doyma süresi; poz bölme (kozmik ışın ve okuma gürültüsü dengesi).
- Tayf için dalga boyu elemanı başına S/N.

## 5. Tayfçeker

- Kırınım ağı denklemi: `m λ = d (sin α + sin β)`; açısal ve doğrusal dağılma, çözme gücü `R = λ/Δλ`, yarık genişliğinin etkisi.
- Echelle düzeni, sıra örtüşmesi, parlama fonksiyonu.
- Aygıt profili; çizgi yayılım fonksiyonu.
- Kalibrasyon lambaları ve dalga boyu çözümü [../05_veri_bilimi/03_fotometri_astrometri_tayf.md](../05_veri_bilimi/03_fotometri_astrometri_tayf.md).

## 6. Gözlem planlama

- Hedef görünürlüğü: yükseklik-zaman eğrisi, hava kütlesi sınırı, meridyen geçişi, alacakaranlık sınırları.
- Ay kısıtı: açısal uzaklık ve aydınlanma oranı.
- Kısıt tabanlı çizelgeleme: öncelik, zaman penceresi, yönlendirme süresi.
- Kalibrasyon planı: bias, karanlık, düz alan (alacakaranlık ya da kubbe), standart yıldızlar (hava kütlesi aralığını kapsayacak biçimde), odak dizisi.

Kâhin: `astroplan` [Morris2018]. PyPI kaydına göre son sürümü Ağustos 2024 tarihlidir; bakım durumu izlenmelidir.

## Test durumları

- Sentetik görüntü üreteci bilinen kazanç ve okuma gürültüsüyle üretir; foton aktarım eğrisi bu değerleri geri verir.
- CCD denklemi üç rejimin limitlerine iner.
- Hava kütlesi formülleri `z < 60°` için `sec z` ile binde bir düzeyinde uyuşur.
- Öngörülen sıfır noktası gerçek bir gözlemevinin yayımlanmış değeriyle karşılaştırılır.
