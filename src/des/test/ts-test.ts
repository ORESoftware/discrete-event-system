

abstract class HasMethod<T =any> {
  abstract implementMe():T
}

class SuperClass extends HasMethod {

  implementMe() {
    return {
      foo: 'bar'
    }
  }
}

class SubClass extends SuperClass {

  implementMe() : any{
    return {
      north: 'star'
    }
  }
}