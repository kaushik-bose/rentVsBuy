import numpy as np
import matplotlib.pyplot as plt


def simulate_rent_vs_buy(
    house_price=1.7e7,
    down_payment=7e6,
    loan_interest=0.085,
    loan_term_years=15,
    rent_initial=52000,
    rent_growth=0.08,
    invest_return_down=0.1,
    house_appreciation=0.072,
    surplus_investment_return=0.12,
    possession_delay_months=0  # Now input is in months
):
    print("\n=== Initial Assumptions ===")
    print(f"House Price: ₹{house_price / 1e7:.2f} Cr")
    print(f"Down Payment: ₹{down_payment / 1e7:.2f} Cr")
    print(f"Loan Interest Rate: {loan_interest * 100:.2f}% p.a.")
    print(f"Loan Term: {loan_term_years} years")
    print(f"Initial Monthly Rent: ₹{rent_initial:,.0f}")
    print(f"Annual Rent Growth Rate: {rent_growth * 100:.2f}%")
    print(f"Investment Return on Down Payment (Renter): {invest_return_down * 100:.2f}% p.a.")
    print(f"House Appreciation Rate: {house_appreciation * 100:.2f}% p.a.")
    print(f"Surplus Investment Return: {surplus_investment_return * 100:.2f}% p.a.\n")

    months = loan_term_years * 12
    r_loan = loan_interest / 12
    r_surplus = surplus_investment_return / 12
    r_down = invest_return_down / 12

    loan_amount = house_price - down_payment
    emi = loan_amount * r_loan * (1 + r_loan)**months / ((1 + r_loan)**months - 1)

    print(f"Monthly EMI: ₹{emi:,.2f}\n")
    if possession_delay_months > 0:
        print(f"Possession Delay: {possession_delay_months} months (Buyer pays rent + EMI during this period)\n")

    rent_schedule = [rent_initial * (1 + rent_growth) ** (i // 12) for i in range(months)]

    buyer_net, renter_net = [], []
    down_invest = down_payment
    surplus_renter = 0
    surplus_buyer = 0
    loan_balance = loan_amount
    crossover_year = None
    crossover_index = None

    for m in range(months):
        rent = rent_schedule[m]

        interest_payment = loan_balance * r_loan
        principal_payment = emi - interest_payment
        loan_balance -= principal_payment

        house_value = house_price * ((1 + house_appreciation) ** (m / 12))
        buyer_equity = house_value - loan_balance if loan_balance > 0 else house_value

        down_invest *= (1 + r_down)

        # --- Possession delay logic ---
        if m < possession_delay_months:
            # Buyer pays both EMI and rent
            buyer_monthly_outflow = emi + rent
            # No surplus for buyer, renter's surplus is as usual
            surplus_buyer = surplus_buyer * (1 + r_surplus)
            surplus_renter = surplus_renter * (1 + r_surplus) + (emi if emi > 0 else 0)
        else:
            # Normal scenario
            if emi > rent:
                monthly_surplus = emi - rent
                surplus_renter = surplus_renter * (1 + r_surplus) + monthly_surplus
                surplus_buyer = surplus_buyer * (1 + r_surplus)
            else:
                monthly_surplus = rent - emi
                surplus_buyer = surplus_buyer * (1 + r_surplus) + monthly_surplus
                surplus_renter = surplus_renter * (1 + r_surplus)

        renter_total = down_invest + surplus_renter
        buyer_total = buyer_equity + surplus_buyer - max(loan_balance, 0)

        renter_net.append(renter_total)
        buyer_net.append(buyer_total)

        if crossover_year is None and renter_total > buyer_total:
            crossover_year = m / 12
            crossover_index = m

    return renter_net, buyer_net, crossover_year, crossover_index, emi


# Run simulation with default values
renter_net, buyer_net, crossover_year, crossover_index, emi = simulate_rent_vs_buy()

print(f"Net Worth at End:")
print(f"Renter: ₹{renter_net[-1] / 1e7:.2f} Cr")
print(f"Buyer: ₹{buyer_net[-1] / 1e7:.2f} Cr")
if crossover_year:
    print(f"\nNet worth crossover occurs at: {crossover_year:.2f} years")

# Plotting
years = np.arange(1, len(renter_net)+1) / 12

# Find intersection (where curves cross)
intersection_year = None
intersection_value = None
for i in range(1, len(renter_net)):
    if (renter_net[i-1] - buyer_net[i-1]) * (renter_net[i] - buyer_net[i]) < 0:
        # Linear interpolation for more accurate intersection
        x0, x1 = (i-1)/12, i/12
        y0_r, y1_r = renter_net[i-1], renter_net[i]
        y0_b, y1_b = buyer_net[i-1], buyer_net[i]
        t = abs(y0_r - y0_b) / (abs(y0_r - y0_b) + abs(y1_r - y1_b))
        intersection_year = x0 + (x1 - x0) * t
        intersection_value = y0_r + (y1_r - y0_r) * t
        break

plt.figure(figsize=(12, 6))
line_renter, = plt.plot(years, renter_net, label="Renter Net Worth")
line_buyer, = plt.plot(years, buyer_net, label="Buyer Net Worth")

if intersection_year is not None:
    plt.axvline(intersection_year, color='gray', linestyle=':', label='Intersection Year')
    plt.axhline(intersection_value, color='gray', linestyle=':')
    plt.scatter([intersection_year], [intersection_value], color='black', zorder=5)
    # Annotate intersection point with default font and color
    plt.annotate(
        f"({intersection_year:.2f} yrs,\n₹{intersection_value/1e7:.2f} Cr)",
        (intersection_year, intersection_value),
        textcoords="offset points", xytext=(10,10), ha='left', fontsize=10
    )
    # Print values on axes
    plt.text(intersection_year, plt.ylim()[0], f"{intersection_year:.2f}", va='bottom', ha='center', fontsize=10)
    plt.text(plt.xlim()[0], intersection_value, f"₹{intersection_value/1e7:.2f} Cr", va='center', ha='left', fontsize=10)

plt.title("Net Worth Over Time: Rent vs Buy")
plt.xlabel("Years")
plt.ylabel("Net Worth (₹)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Extend axis limits for annotation padding ---
ax = plt.gca()
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
x_pad = (x_max - x_min) * 0.10  # Increased from 0.04 to 0.10
y_pad = (y_max - y_min) * 0.18  # Increased from 0.08 to 0.18
ax.set_xlim(x_min - x_pad, x_max + x_pad)
ax.set_ylim(y_min - y_pad, y_max + y_pad)

# --- Interactivity: Show values on hover ---
fig = plt.gcf()
ax = plt.gca()
annot = ax.annotate(
    "", xy=(0,0), xytext=(20,20), textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->")
)
annot.set_visible(False)

def update_annot(ind, xdata, ydata_r, ydata_b):
    x = xdata[ind]
    y_r = ydata_r[ind]
    y_b = ydata_b[ind]
    annot.xy = (x, (y_r + y_b) / 2)
    text = f"Year: {x:.2f}\nRenter: ₹{y_r/1e7:.2f} Cr\nBuyer: ₹{y_b/1e7:.2f} Cr"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.9)

def hover(event):
    if event.inaxes == ax:
        xdata = years
        ydata_r = np.array(renter_net)
        ydata_b = np.array(buyer_net)
        # Find nearest index
        if event.xdata is not None:
            idx = np.searchsorted(xdata, event.xdata)
            if idx > 0 and (idx == len(xdata) or abs(event.xdata - xdata[idx-1]) < abs(event.xdata - xdata[idx])):
                idx -= 1
            if 0 <= idx < len(xdata):
                update_annot(idx, xdata, ydata_r, ydata_b)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
    annot.set_visible(False)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
