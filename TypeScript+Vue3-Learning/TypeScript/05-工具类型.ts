// 工具类型（Utility Types）是内置的一组泛型类型，用来在类型系统中进行类型变换和重构。它们广泛用于构建更复杂的类型，提升类型的灵活性和复用性。

type User = {
  id: number,
  name: string,
  age: number;
  email?: string // ?表示这个属性是可选的
};

// 常用的工具类型

// 1、Partial。所有属性变为可选
type PartialUser = Partial<User>; // { id?: number; name?: string; age?: number; email?: string }

// 2、Required。所有属性都变为必选
type StrictUser = Required<User>; // { id: number; name: string; age: number; email: string }

// 3、所有属性都变为只读
type ReadonlyUser = Readonly<User>; // 所有字段不可更改。
// 用 const 修饰的变量可以更改字段的值，但变量不能重新赋值为另一个值
// 用 Readonly 规定的类型不可以更改字段的值，但变量可以重新赋值为另一个值

// 4、Pick。挑选出几个属性
type UserPreview = Pick<User, 'id' | 'name'>; // { id: number; name: string }

// 5、去掉几个属性
type SafeUser = Omit<User, 'email'>; // { id: number; name: string; age: number }

// 6、构造键值统一的对象类型
type Role = 'admin' | 'editor' | 'guest';
type RolePermissions = Record<Role, boolean>; // { admin: boolean; editor: boolean; guest: boolean }
