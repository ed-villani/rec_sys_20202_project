from readers.Recipes import NutrientsReader
import numpy as np
import csv


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
all_data = []
for i, (id, name, preparation, ingredients, nutrients) in enumerate(reader.read()):
    den = get_nutrition_density(nutrients)
    densities.append(den)
    recipe2density[id] = get_nutrition_density(nutrients)
    all_data.append([id, name, preparation, ingredients])


arr = np.array(densities)
x = arr.argsort()[-int(len(all_data) / 5) :][::-1]

print(all_data[x[-1]])
print(all_data[x[-2]])
print(all_data[x[-3]])
print(all_data[x[-4]])
print(all_data[x[-5]])
print(all_data[x[-6]])

with open("healthy_recipes_20.csv", "w") as file:
    writer = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["recipe_id", "recipe_name", "ingredients", "cooking_directions"])
    for index in x:
        writer.writerow(all_data[index])
