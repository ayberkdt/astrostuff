# Işınım ve fotometri fiziği

Hedef modüller: `isinim/`, `fotometri/kalibrasyon.py`.

Denklemler planlama düzeyindedir; kodlanmadan önce anılan kaynaktan okunur.

## 1. Işınım nicelikleri

- Özgül yeğinlik `I_ν` [W m⁻² Hz⁻¹ sr⁻¹], akı yoğunluğu `F_ν`, toplam akı `F`, ışınım gücü `L` [RybickiLightman1979, Böl. 1].
- `F_ν` ile `F_λ` dönüşümü: `F_ν = (λ²/c) F_λ`.
- Ters kare yasası: `F = L / (4π d²)`.
- Jansky: `1 Jy = 1e-26 W m⁻² Hz⁻¹`.
- Yüzey parlaklığı: yakın evrende uzaklıktan bağımsızdır; kozmolojik uzaklıkta `(1+z)⁻⁴` ile söner.

## 2. Kara cisim

- Planck fonksiyonu: `B_ν(T) = (2hν³/c²) / (exp(hν/kT) − 1)` ve `B_λ` biçimi [RybickiLightman1979, Böl. 1].
- Wien kayma yasası, Stefan-Boltzmann yasası `F = σT⁴`, Rayleigh-Jeans ve Wien limitleri.
- Sayısal kararlılık: küçük `x = hν/kT` için `expm1`, büyük `x` için taşma denetimi.
- Etkin sıcaklık, renk sıcaklığı ve parlaklık sıcaklığı tanımları [CarrollOstlie2017, Böl. 3].
- Sabitler: `h, c, k, σ` CODATA önerilen değerlerinden [CODATA2022].

## 3. Kadir sistemleri

- Pogson tanımı: `m1 − m2 = −2,5 log10(F1/F2)`.
- **AB sistemi:** `m_AB = −2,5 log10(F_ν / 3631 Jy)`; düz `F_ν` tayfı her bantta aynı kadiri verir [OkeGunn1983].
- **Vega sistemi:** sıfır noktası Vega'nın (ya da modelinin) tayfıdır; bant başına AB-Vega farkı tablolanır [BessellMurphy2012].
- **ST sistemi:** düz `F_λ` referansı.
- Mutlak kadir ve uzaklık modülü: `m − M = 5 log10(d / 10 pc) + A + K`.
- Bolometrik kadir: sıfır noktası IAU 2015 B2 kararıyla sabitlenmiştir; `M_bol = 0` değeri `L = 3,0128e28 W` ışınım gücüne karşılık gelir [Mamajek2015]. Güneş'in nominal değerleri IAU 2015 B3'ten [Prsa2016].
- Kadir belirsizliği: `σ_m ≈ 1,0857 · σ_F / F`; düşük S/N'de asimetri ve asinh kadir.

## 4. Filtreler ve sentetik fotometri

- Bant geçiren eğrisi `S(λ)`: dedektör foton sayar, dolayısıyla integral foton ağırlıklıdır:
  `⟨F_λ⟩ = ∫ F_λ S λ dλ / ∫ S λ dλ` [BessellMurphy2012, Ek A].
- Enerji ve foton karşılıklı filtre eğrilerinin ayrımı yaygın bir hata kaynağıdır; her filtre dosyasının türü üstveride tutulur.
- Etkin, pivot ve ortalama dalga boyu tanımları.
- Sistemler: Johnson-Cousins UBVRI [Bessell1990; Bessell2005], SDSS ugriz [Fukugita1996], Gaia G/BP/RP, 2MASS JHK [Skrutskie2006].
- Renk dönüşümleri: sistemler arası deneysel bağıntılar; geçerlilik renk aralığı kaydedilir.
- Standart yıldızlar: Landolt alanları [Landolt1992].

## 5. Yıldızlararası sönümleme ve kızarma

- `A_λ` sönümleme, `E(B−V) = A_B − A_V` renk artığı, `R_V = A_V / E(B−V)`; Samanyolu ortalaması yaklaşık 3,1.
- Sönümleme yasaları: `A_λ/A_V` eğrisinin `R_V` ile parametrelenmesi [CCM1989; Fitzpatrick1999].
- Toz haritaları: tüm gökyüzü `E(B−V)` haritası [SFD1998] ve yeniden kalibrasyonu [SchlaflyFinkbeiner2011]. Haritanın çözünürlüğü ve düşük galaktik enlemdeki güvenilmezliği belgelenir.
- Wesenheit kadiri: kızarmadan bağımsız birleşim `W = m − R (renk)` [Madore1982]; Sefe uzaklık merdiveninde kullanılır.

## 6. K düzeltmesi

Kırmızıya kaymış kaynağın gözlenen banttaki akısı, durgun çerçevedeki başka bir dalga boyu aralığından gelir. Düzeltme tayfın biçimine ve bant çiftine bağlıdır; tanım ve türetim [Hogg2002; OkeSandage1968]. Uzaklık modülündeki `K` terimi budur.

## 7. Işınım aktarımı

- Aktarım denklemi: `dI_ν/dτ_ν = −I_ν + S_ν`; optik derinlik, kaynak fonksiyonu [RybickiLightman1979, Böl. 1].
- Optik olarak ince ve kalın limitler; düzlem-paralel atmosferde biçimsel çözüm.
- Kenar kararması: doğrusal, karesel ve doğrusal olmayan yasalar; geçiş ışık eğrisi modelinin girdisidir [MandelAgol2002].
- Çizgi profilleri: Doppler (Gauss), doğal ve basınç genişlemesi (Lorentz), Voigt profili; eşdeğer genişlik ve büyüme eğrisi.
- Saha ve Boltzmann denklemleri: iyonlaşma ve uyarılma dengesi [CarrollOstlie2017, Böl. 8].

## 8. Işınım süreçleri (ileri aşama)

Serbest-serbest ışınım, sinkrotron, Compton saçılması, toz ışınımı (değiştirilmiş kara cisim) [RybickiLightman1979, Böl. 5-7; Draine2011]. Tayfsal enerji dağılımı uydurma projesinin bileşenleridir.

## Test durumları

- Planck fonksiyonunun integrali `σT⁴/π` vermelidir; tepe dalga boyu Wien sabitiyle uyuşmalıdır.
- Vega ve Güneş tayfının sentetik fotometrisi yayımlanmış kadirlerle karşılaştırılır [BessellMurphy2012].
- `synphot` ve `speclite` ile aynı tayf ve filtrede kadir farkı `1e-4` mag altında.
- Sönümleme yasaları `dust_extinction` ve `extinction` paketleriyle karşılaştırılır.
