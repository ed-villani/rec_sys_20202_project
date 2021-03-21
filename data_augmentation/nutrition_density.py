from readers.Recipes import NutrientsReader
import numpy as np


def get_nutrition_density(
    data,
    bad_set=set(
        [
            "sodium",
            "sugars",
            "calories",
            "saturatedFat",
            "carbohydrates",
            "fat",
            "cholesterol",
        ]
    ),
):
    penalizer = 0
    nutrients = 0
    for key, info in data.items():
        try:
            dv = float(info["percentDailyValue"])
            if key in bad_set:
                penalizer += dv
            else:
                nutrients += dv
        except ValueError:
            pass
        except TypeError:
            pass
    return 0 if penalizer == 0 else nutrients / penalizer


reader = NutrientsReader("inputs/core-data_recipe.csv")
recipe2density = {}
id2nutrient = {}
densities = []
all_nutrients = []
all_names = []
for i, (id, nutrients) in enumerate(reader.read()):
    den = get_nutrition_density(nutrients)
    densities.append(den)
    recipe2density[id] = get_nutrition_density(nutrients)


arr = np.array(densities)
print(arr.mean())
best = all_names[arr.argmax()]
best_n = all_nutrients[arr.argmax()]
print(all_nutrients[arr.argmax()])
