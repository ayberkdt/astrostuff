# Görelilik, sıkışık cisimler ve yüksek enerji astrofiziği

Hedef modüller (ileri aşama): `gorelilik/`, `sikisik/`, `cekimsel_dalga/`.

Bu bölüm yol haritasının son aşamalarına aittir. Amaç bu alanların kodlarını yeniden yazmak değil, kapalı biçimli ya da tek boyutlu integralle çözülen temel sonuçları kaynaklı ve sınanmış fonksiyonlar olarak toplamaktır. Denklemler planlama düzeyindedir; kodlanmadan önce anılan kaynaktan okunur.

## 1. Özel görelilik

- Lorentz dönüşümü, hız toplama, Lorentz çarpanı.
- Görelilikli Doppler kayması ve [Hogg1999, Denk. 9] ile bağlantısı: `1 + z = sqrt((1 + v/c)/(1 − v/c))`. Bu denklem makalenin TeX kaynağından okunmuştur.
- Aberasyon ve ışınım demetlenmesi; Doppler çarpanı `δ = 1/[γ(1 − β cos θ)]`; görünür ışık üstü hareket [RybickiLightman1979, Böl. 4].
- Kırmızıya kaymaların birleştirilmesi: `(1 + z_göz) = (1 + z_koz)(1 + z_özel)`; küçük hızlarda [Hogg1999, Denk. 10]. Süpernova analizinde güneş merkezli ve CMB çerçeveli kırmızıya kaymaların ayrımı buradan gelir [Carr2022].

## 2. Genel görelilik: küresel simetrik uzay-zaman

- Schwarzschild metriği, Schwarzschild yarıçapı `r_s = 2GM/c²`, çekimsel kırmızıya kayma `1 + z = (1 − r_s/r)^(−1/2)` [Carroll2019, Böl. 5].
- Jeodezikler: etkin potansiyel, en içteki kararlı dairesel yörünge (`3 r_s`), foton küresi (`1,5 r_s`), kara delik gölgesinin açısal boyutu.
- Klasik sınamalar: günberi ilerlemesi `Δϖ = 6πGM/[c² a(1 − e²)]` (dönem başına), ışığın sapması `4GM/(c² b)`, Shapiro gecikmesi [Carroll2019, Böl. 5]. Zaman ve koordinat modüllerindeki düzeltmelerin kaynağıdır.
- Kerr metriği: dönme parametresi, ergosfer, yörünge yarıçapının dönmeye bağlılığı (yalnızca kapalı biçimler).

## 3. Sıkışık cisimler

- Beyaz cüceler: yozlaşmış elektron gazının durum denklemi, kütle-yarıçap bağıntısı, Chandrasekhar kütlesi [ShapiroTeukolsky1983, Böl. 3].
- Nötron yıldızları: Tolman-Oppenheimer-Volkoff denklemi; verilen bir durum denklemi için kütle-yarıçap eğrisinin sayısal integrali [ShapiroTeukolsky1983, Böl. 9].
- Yığılma: Eddington ışınım gücü `L_Edd = 4πGMm_p c/σ_T`, yığılma verimi, ince disk sıcaklık profili ve çok renkli kara cisim tayfı [Longair2011, Böl. 14]. `σ_T` ve `m_p` sabitler modülünde CODATA 2022 değerleriyle tanımlıdır [CODATA2022].
- Pulsarlar: manyetik dipol frenlemesi, karakteristik yaş, dağılma ölçüsü ve geliş zamanı gecikmesi `Δt ∝ DM ν⁻²`.

## 4. Çekimsel dalgalar

- Dörtlü kutup formülü; iki cisimli sistemin ışıdığı güç ve yörünge küçülmesi [Peters1964]. Hulse-Taylor pulsarının dönem azalması bu formülün sınamasıdır.
- Cıvıltı kütlesi `𝓜 = (m1 m2)^(3/5)/(m1 + m2)^(1/5)`; frekans evrimi `ḟ ∝ 𝓜^(5/3) f^(11/3)`; birleşmeye kalan süre [Maggiore2007, Böl. 4].
- Dalga biçimi: Newton mertebesinde genlik ve faz; post-Newton düzeltmeler yeniden yazılmaz [Blanchet2014].
- Gerinim genliğinden ışınım gücü uzaklığı: standart siren; kırmızıya kaymayla birleşince `H0` verir [Abbott2017].
- Eşleşmiş süzgeçle arama: gürültü güç tayfı, beyazlatma, şablon bankası, S/N zaman serisi [Allen2012].
- Uçtan uca hedef: GW150914 açık verisinde sinyalin beyazlatma ve bant geçiren süzgeçle görünür kılınması, tek şablonla eşleşmiş süzgeç S/N'si [Abbott2016]. Kâhinler: `gwpy`, `pycbc`.

## 5. Yüksek enerji ışınım süreçleri

- Sinkrotron: tek elektronun tayfı, kuvvet yasası elektron dağılımının tayf indisi `α = (p − 1)/2`, öz soğurma [RybickiLightman1979, Böl. 6].
- Ters Compton saçılması: Thomson ve Klein-Nishina rejimleri; sinkrotron öz-Compton [RybickiLightman1979, Böl. 7].
- Isıl frenleme ışınımı: yayma katsayısı, Gaunt çarpanı; gökada kümelerinin X-ışını ışınımı [RybickiLightman1979, Böl. 5].
- Sunyaev-Zeldovich etkisi: Compton `y` parametresi (ileri aşama).
- Kozmik ışın ve gama ışını soğurması: çift üretimi ve gökada dışı arka plan ışığı (ileri aşama) [Longair2011].

## 6. Sayım istatistiği

Yüksek enerji verisi foton listesidir; kutu başına sayım düşüktür.

- Poisson olabilirliği ve Cash istatistiği [Cash1979].
- Açık-kapalı ölçümde anlamlılık; arka plan kestiriminin belirsizliği.
- Üst sınırlar: Bayesçi ve sıklıkçı yapımlar; kapsama denetimi.
- Aygıt yanıtı: etkin alan, enerji dağılma matrisi, PSF; ileri katlama (modelin yanıtla evriştirilip sayımlarla karşılaştırılması).

Kâhinler: `gammapy`, `sherpa`.

## Doğrulama

- Merkür'ün günberi ilerlemesi yüzyılda yaklaşık 43 yay saniyesi verir (kaynak değeri kodlama sırasında okunacak).
- Güneş kenarında ışık sapması 1,75 yay saniyesi dolayındadır.
- Chandrasekhar kütlesi ve TOV çözümünün sınır kütlesi kaynak değerleriyle karşılaştırılır.
- Peters formülünün Hulse-Taylor sistemi için verdiği dönem azalması yayımlanan ölçümle karşılaştırılır.
- Cıvıltı kütlesinden türetilen frekans evrimi GW150914'ün yayımlanan parametreleriyle tutarlıdır.
