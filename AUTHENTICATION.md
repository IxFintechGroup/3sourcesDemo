# Authentication System

This application now includes a simple user authentication system to protect access to the Multi-Source OHLCV reconciliation tool.

## Features

- **Login Page**: Users must authenticate before accessing the main application
- **Session Management**: Authentication state is maintained throughout the session
- **Logout Functionality**: Users can log out from the sidebar
- **Secure Password Storage**: Passwords are hashed using SHA-256
- **Demo Accounts**: Pre-configured accounts for testing

## Demo Accounts

The following demo accounts are available for testing:

| Username | Password | Description |
|----------|----------|-------------|
| `admin`  | `password` | Administrator account |
| `user`   | `123456`   | Regular user account |
| `demo`   | `hello`    | Demo account |
| `guest`  | (empty)    | Guest account with no password |

## How to Use

1. **Start the Application**: Run `streamlit run streamlit_app.py`
2. **Login**: Enter your username and password on the login page
3. **Access the App**: Once authenticated, you'll have full access to the OHLCV reconciliation features
4. **Logout**: Click the logout button in the sidebar when you're done

## Security Notes

- **Development Only**: This is a simple authentication system suitable for development and demo purposes
- **Production Use**: For production environments, consider implementing:
  - Database-backed user management
  - Environment variable configuration
  - More robust password policies
  - Session timeout mechanisms
  - HTTPS enforcement
  - Rate limiting for login attempts

## Configuration

User credentials are stored in `auth_config.py`. To add new users:

1. Hash the password using SHA-256
2. Add the username and hashed password to the `VALID_USERS` dictionary
3. Update the `get_demo_credentials()` function if needed

Example:
```python
import hashlib
password = "mypassword"
hashed = hashlib.sha256(password.encode()).hexdigest()
print(hashed)  # Use this hash in VALID_USERS
```

## Files Modified

- `streamlit_app.py`: Added authentication logic and login page
- `auth_config.py`: Created separate configuration file for user credentials
- `AUTHENTICATION.md`: This documentation file
