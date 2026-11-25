# Gökadalar ve dinamik

Hedef modül: `gokada/`.

Denklemler planlama düzeyindedir; kodlanmadan önce anılan kaynaktan okunur.

## 1. Çekim potansiyeli kuramı

- Poisson denklemi: `∇²Φ = 4πGρ`; küresel sistemde Newton teoremleri; dairesel hız `v_c² = r dΦ/dr`; kaçış hızı [BinneyTremaine2008, Böl. 2].
- Analitik potansiyel-yoğunluk çiftleri:
  - nokta kütle, homojen küre
  - Plummer küresi [Plummer1911]
  - Hernquist profili [Hernquist1990]
  - Navarro-Frenk-White profili: `ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]`; yoğunlaşma parametresi, virial kütle ve yarıçap tanımları [NFW1997]
  - tekil ve çekirdekli izotermal küre, logaritmik potansiyel
  - disk potansiyelleri: Miyamoto-Nagai, üstel disk (Bessel fonksiyonlu dönme eğrisi) [BinneyTremaine2008, Böl. 2]
- Her profil için tek arayüz: yoğunluk, potansiyel, kuvvet, çevrelenen kütle, dairesel hız. Bileşik potansiyel bu arayüzün toplamıdır.

Kâhinler: `galpy` ve `gala`.

## 2. Yörüngeler

- Küresel ve eksenel simetrik potansiyelde yörünge integrasyonu; simplektik integratör [../03_matematik/03_sayisal_yontemler.md](../03_matematik/03_sayisal_yontemler.md).
- Hareket integralleri: enerji, `L_z`; etkin potansiyel, episiklik yaklaşım, Oort sabitleri [BinneyTremaine2008, Böl. 3].
- Galaktosentrik çerçeve: Güneş'in merkeze uzaklığı, yükseklik ve özel hızı; ICRS + uzaklık + öz hareket + dikine hızdan 6 boyutlu konum-hız.

## 3. Dönme eğrileri ve karanlık madde

- Gözlenen dönme eğrisinin bileşenlere ayrılması: `v_c² = v_disk² + v_gaz² + v_şişkinlik² + v_hale²`.
- Kütle-ışınım gücü oranı ile hale parametreleri arasındaki dejenerasyon.
- Tully-Fisher bağıntısı: dönme hızı ile ışınım gücü arasındaki kuvvet yasası; uzaklık göstergesi [TullyFisher1977].
- Virial teoremi ve kütle kestirimi: `M ≈ σ² R / G` mertebesi; izdüşüm ve anizotropi katsayıları [BinneyTremaine2008, Böl. 4].

## 4. Çarpışmasız dinamik

- Çarpışmasız Boltzmann denklemi ve Jeans denklemleri; küresel Jeans denklemiyle hız dağılımından kütle profili [BinneyTremaine2008, Böl. 4].
- Dağılım fonksiyonları, Jeans teoremi; analitik modellerden başlangıç koşulu üretme.
- Gevşeme süresi, dinamik sürtünme (Chandrasekhar formülü), gelgit yarıçapı [BinneyTremaine2008, Böl. 7-8].

## 5. Işınım profilleri ve yapı ölçümü

- Sérsic profili: `I(R) = I_e exp{−b_n [(R/R_e)^(1/n) − 1]}`; `b_n` katsayısı, toplam akı, Petrosian ve Kron nicelikleri [GrahamDriver2005; Kron1980].
- Üstel disk ve de Vaucouleurs profili Sérsic ailesinin `n = 1` ve `n = 4` üyeleridir.
- İki boyutlu profil uydurma: PSF evrişimi, eliptiklik ve konum açısı; görüntü modellemesi.
- Morfoloji ölçüleri: yoğunlaşma, asimetri, Gini katsayısı (ileri aşama).

## 6. Büyük ölçekli yapı (ileri aşama)

- İki nokta ilinti fonksiyonu ve Landy-Szalay kestiricisi; güç tayfı [LandySzalay1993].
- Baryon akustik salınımları: ilinti fonksiyonunda ses ufku ölçeği [Eisenstein2005]; aktarım fonksiyonu için uydurma formülü [EisensteinHu1998].
- Hale kütle fonksiyonu ve hale modeli [MoBoschWhite2010, Böl. 7].

## 7. Çekimsel merceklenme (ileri aşama)

Mercek denklemi, Einstein yarıçapı, nokta ve tekil izotermal küre mercekleri, büyütme; zaman gecikmesi ve Hubble sabitine bağlantısı [MoBoschWhite2010, Böl. 6].

## Test durumları

- Her potansiyel için `∇²Φ = 4πGρ` sayısal türevle doğrulanır; `r → ∞` limitinde toplam kütle.
- Dairesel yörünge bir dönem sonunda kapanır; enerji hatası sınırlı kalır.
- `galpy` ile aynı potansiyelde kuvvet ve dairesel hız `1e-10` bağıl farkla uyuşur.
- Sérsic toplam akısının sayısal integrali kapalı biçimle uyuşur [GrahamDriver2005].
